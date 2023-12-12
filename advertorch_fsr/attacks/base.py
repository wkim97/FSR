from abc import ABCMeta

import torch

from ..utils import replicate_input


class Attack(object):

    __metaclass__ = ABCMeta

    def __init__(self, predict, loss_fn, clip_min, clip_max):
        self.predict = predict
        self.loss_fn = loss_fn
        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, x, **kwargs):
        error = "Sub-classes must implement perturb."
        raise NotImplementedError(error)

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)


class LabelMixin(object):
    def _get_predicted_label(self, x):
        with torch.no_grad():
            outputs, _, _, _ = self.predict(x, is_eval=True)
        _, y = torch.max(outputs, dim=1)
        return y

    def _verify_and_process_inputs(self, x, y):
        if self.targeted:
            assert y is not None

        if not self.targeted:
            if y is None:
                assert False, 'None target'

        x = replicate_input(x)
        return x, y
