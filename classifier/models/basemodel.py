import torch.nn as nn
import abc


class BaseModel(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self, num_classes=None, **kwargs):
        assert num_classes is not None, 'BaseModel received None value'
        super().__init__()
        self.num_classes = num_classes

    @abc.abstractmethod
    def forward(self, sample) -> dict:
        """forward method, must implement"""
        raise NotImplementedError('users must define forward to use this base class')

    @abc.abstractmethod
    def get_forward_output_keys(self) -> list:
        """return the keys of the dict returned by forward, must implement"""
        raise NotImplementedError('users must define get_forward_output_keys to use this base class')

    @abc.abstractmethod
    def get_forward_input_keys(self) -> list:
        """return the keys of the dict needed by forward, must implement"""
        raise NotImplementedError('users must define get_forward_input_keys to use this base class')

