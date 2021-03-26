import torch.nn as nn
import numpy as np
import torch
import abc


class BaseLoss(nn.Module):
    __metaclass__ = abc.ABC

    def __init__(self, cfg, **kwargs):
        super(BaseLoss, self).__init__()
        self.cfg = cfg

    def forward(self, input, target):
        assert isinstance(input, dict), 'input is not a dictionary'
        assert isinstance(target, dict), 'input is not a dictionary'

        for key in self.get_required_keys():
            i, t = input[key], target[key]
            if isinstance(i, (list, np.ndarray)):
                i = torch.as_tensor(np.array(i))
            if isinstance(t, (list, np.ndarray)):
                t = torch.as_tensor(np.array(t))
            t = t.to(device=i.device)
            input[key] = i
            target[key] = t

        return input, target

    @abc.abstractmethod
    def get_required_keys(self):
        raise NotImplementedError('users must define get_required_keys to use this base class')
