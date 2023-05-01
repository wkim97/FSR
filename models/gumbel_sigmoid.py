import torch
from torch import nn


class GumbelSigmoid(nn.Module):
    def __init__(self, tau=1.0):
        super(GumbelSigmoid, self).__init__()

        self.tau = tau
        self.softmax = nn.Softmax(dim=1)
        self.p_value = 1e-8

    def forward(self, x, is_eval=False):
        r = 1 - x

        x = (x + self.p_value).log()
        r = (r + self.p_value).log()

        if not is_eval:
            x_N = torch.rand_like(x)
            r_N = torch.rand_like(r)
        else:
            x_N = 0.5 * torch.ones_like(x)
            r_N = 0.5 * torch.ones_like(r)

        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()

        x = x + x_N
        x = x / (self.tau + self.p_value)
        r = r + r_N
        r = r / (self.tau + self.p_value)

        x = torch.cat((x, r), dim=1)
        x = self.softmax(x)

        return x
