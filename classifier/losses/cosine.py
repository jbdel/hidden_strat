import torch.nn as nn
import torch
from .baseloss import BaseLoss


class CosineLoss(BaseLoss):
    def __init__(self, cfg, **kwargs):
        super(CosineLoss, self).__init__(cfg, **kwargs)
        self.func = nn.CosineEmbeddingLoss(reduction="sum")

    def forward(self, input, target):
        input, target = super().forward(input, target)
        input, target = input['vector'], target['vector']
        index = torch.ones(input.size()[0]).to(device=input.device)
        return self.func(input, target, index)

    def get_required_keys(self):
        return ['vector']
