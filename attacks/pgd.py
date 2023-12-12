'''
Codes modified from original codes of:
CAS : https://github.com/bymavis/CAS_ICLR2021
'''


import torch
import torch.nn as nn


def project(x, original_x, epsilon, _type='linf'):
    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)

    else:
        raise NotImplementedError

    return x


class PGD():
    def __init__(self, model, epsilon, alpha, min_val, max_val, max_iters, _type='linf'):
        self.model = model

        self.epsilon = epsilon
        self.alpha = alpha
        self.min_val = min_val
        self.max_val = max_val
        self.max_iters = max_iters
        self._type = _type

    def perturb(self, original_images, labels, random_start=False):
        device = original_images.get_device()

        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.cuda(device)
            x = original_images + rand_perturb
            x.clamp_(self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True

        with torch.enable_grad():
            for _iter in range(self.max_iters):
                outputs, _, _, _ = self.model(x, is_eval=True)
                cls_loss = nn.CrossEntropyLoss()(outputs, labels)

                loss = cls_loss
                grad_outputs = None
                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs,
                                            only_inputs=True)[0]
                x.data += self.alpha * torch.sign(grads.data)
                x = project(x, original_images, self.epsilon, self._type)
                x.clamp_(self.min_val, self.max_val)

        return x
