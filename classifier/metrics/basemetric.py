import torch.nn as nn
import numpy as np
import torch
import abc


class BaseMetric(nn.Module):
    __metaclass__ = abc.ABC

    def __init__(self, cfg, **kwargs):
        super(BaseMetric, self).__init__()
        self.cfg = cfg

    def forward(self, input, target):
        assert isinstance(input, dict), 'input is not a dictionary'
        assert isinstance(target, dict), 'target is not a dictionary'

        input = {k: np.array(v) for k, v in input.items()}
        target = {k: np.array(v) for k, v in target.items()}

        return input, target

    @abc.abstractmethod
    def get_required_keys(self):
        raise NotImplementedError('users must define get_required_keys to use this base class')
