from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn

from ..utils import clamp
from ..utils import normalize_by_pnorm
from ..utils import clamp_by_pnorm
from ..utils import is_float_or_torch_tensor
from ..utils import batch_multiply
from ..utils import batch_clamp
from ..utils import replicate_input
from ..utils import batch_l1_proj

from .base import Attack
from .base import LabelMixin
from .utils import rand_init_delta


def perturb_iterative(xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0,
                      l1_sparsity=None):

    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    delta.requires_grad_()
    for ii in range(nb_iter):
        outputs, _, _, _ = predict(xvar + delta, is_eval=True)
        loss = loss_fn(outputs, yvar)

        if minimize:
            loss = -loss

        loss.backward()
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data

        elif ord == 2:
            grad = delta.grad.data
            grad = normalize_by_pnorm(grad)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)

        elif ord == 1:
            grad = delta.grad.data
            abs_grad = torch.abs(grad)

            batch_size = grad.size(0)
            view = abs_grad.view(batch_size, -1)
            view_size = view.size(1)
            if l1_sparsity is None:
                vals, idx = view.topk(1)
            else:
                vals, idx = view.topk(
                    int(np.round((1 - l1_sparsity) * view_size)))

            out = torch.zeros_like(view).scatter_(1, idx, vals)
            out = out.view_as(grad)
            grad = grad.sign() * (out > 0).float()
            grad = normalize_by_pnorm(grad, p=1)
            delta.data = delta.data + batch_multiply(eps_iter, grad)

            delta.data = batch_l1_proj(delta.data.cpu(), eps)
            if xvar.is_cuda:
                delta.data = delta.data.cuda()
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)
        delta.grad.data.zero_()

    x_adv = clamp(xvar + delta, clip_min, clip_max)
    return x_adv


class PGDAttack(Attack, LabelMixin):

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, l1_sparsity=None, targeted=False):

        super(PGDAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.l1_sparsity = l1_sparsity
        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(
                delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        rval = perturb_iterative(
            x, y, self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            l1_sparsity=self.l1_sparsity
        )

        return rval.data


class LinfPGDAttack(PGDAttack):

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False):
        ord = np.inf
        super(LinfPGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted,
            ord=ord)
