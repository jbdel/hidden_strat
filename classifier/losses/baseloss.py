import torch.nn as nn
import numpy as np
import torch
import abc


class BaseLoss(nn.Module):
    __metaclass__ = abc.ABC

    def __init__(self, cfg, weight=1.0, **kwargs):
        super(BaseLoss, self).__init__()
        self.cfg = cfg
        self.weight = weight

        self.iteration = 0
        self.running_loss = 0
        self.mean_running_loss = 0

    def forward(self, input, target):
        if isinstance(input, (list, np.ndarray)):
            input = torch.as_tensor(np.array(input))
        if isinstance(target, (list, np.ndarray)):
            target = torch.as_tensor(np.array(target))
        return input, target.to(device=input.device)

    def update_running_loss(self, loss):
        self.iteration += 1
        self.running_loss += loss.item()
        self.mean_running_loss = self.running_loss / self.iteration
        # self.mean_running_loss = loss.item()
