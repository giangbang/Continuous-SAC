# Batch Renormalization for convolutional neural nets implementation based
# on https://arxiv.org/abs/1702.03275

from torch.nn import Module
import torch
import numpy as np


def val_schel(init_val, final_val, init_step, final_step, curr_step):
    if curr_step >= final_step:
        return final_val
    elif curr_step <= init_step:
        return init_val

    frac = (curr_step - init_step) / (final_step - init_step)
    return (final_val - init_val) * frac + init_val


class BatchRenormalization(Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.995):
        super().__init__()

        self.eps = eps
        self.momentum = torch.nn.Parameter(torch.tensor(momentum, requires_grad=False))

        self.gamma = torch.nn.Parameter(
            torch.ones((1, num_features)), requires_grad=True
        )
        self.beta = torch.nn.Parameter(
            torch.zeros((1, num_features)), requires_grad=True
        )

        self.ra_mean = torch.zeros((1, num_features), requires_grad=False)

        self.ra_var = torch.ones((1, num_features), requires_grad=False)

        self.max_r_max = 2.0
        self.max_d_max = 2.0

        self.init_r_max = 1.0
        self.init_d_max = 0.0

        self.steps = 0

    def to(self, device):
        super().to(device)
        self.ra_mean = self.ra_mean.to(device)
        self.ra_var = self.ra_var.to(device)
        self.momentum = self.momentum.to(device)

    def forward(self, x):
        device = self.gamma.device
        self.ra_mean = self.ra_mean.to(device)
        self.ra_var = self.ra_var.to(device)
        if self.training:
            var, mean = torch.var_mean(x, dim=0, keepdim=True)

            # r_max = val_schel(self.init_r_max, self.max_r_max, 5000, 40000, self.steps)
            # d_max = val_schel(self.init_d_max, self.max_d_max, 5000, 25000, self.steps)
            r_max = 3
            d_max = 5

            std = torch.sqrt(var + self.eps)
            ra_std = torch.sqrt(self.ra_var + self.eps)
            r = (std / ra_std).detach()
            r = torch.clamp(r, 1 / r_max, r_max)
            d = ((mean - self.ra_mean) / ra_std).detach()
            d = torch.clamp(d, -d_max, d_max)

            self.ra_mean = (
                self.momentum * self.ra_mean + (1 - self.momentum) * mean.detach()
            )
            self.ra_var = (
                self.momentum * self.ra_var + (1 - self.momentum) * var.detach()
            )
            self.steps += 1

            x = ((x - mean) * r) / std + d
            x = self.gamma * x + self.beta
        else:
            x = (x - self.ra_mean.detach()) / torch.sqrt(self.ra_var.detach())
            x = self.gamma * x + self.beta
        return x
