from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn

from ..utils import clamp
from ..utils import normalize_by_pnorm
from ..utils import batch_multiply

from .base import Attack
from .base import LabelMixin


class GradientSignAttack(Attack, LabelMixin):

    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0.,
                 clip_max=1., targeted=False):
        super(GradientSignAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):

        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.requires_grad_()
        outputs, _, _, _ = self.predict(xadv, is_eval=True)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        grad_sign = xadv.grad.detach().sign()

        xadv = xadv + batch_multiply(self.eps, grad_sign)

        xadv = clamp(xadv, self.clip_min, self.clip_max)

        return xadv.detach()


FGSM = GradientSignAttack


class GradientAttack(Attack, LabelMixin):

    def __init__(self, predict, loss_fn=None, eps=0.3,
                 clip_min=0., clip_max=1., targeted=False):
        super(GradientAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.requires_grad_()
        outputs, _, _, _ = self.predict(xadv, is_eval=True)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        grad = normalize_by_pnorm(xadv.grad)
        xadv = xadv + batch_multiply(self.eps, grad)
        xadv = clamp(xadv, self.clip_min, self.clip_max)

        return xadv.detach()


FGM = GradientAttack
