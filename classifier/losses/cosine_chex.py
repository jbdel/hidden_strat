import torch.nn as nn
import torch
from .baseloss import BaseLoss


class CosineLossChex(BaseLoss):
    def __init__(self, cfg, **kwargs):
        super(CosineLossChex, self).__init__(cfg, **kwargs)
        self.func = nn.CosineEmbeddingLoss(reduction="mean")

    def forward(self, input, target):
        input, target = input['vector'], input['chex_vector']
        input, target = super().forward(input, target)
        index = torch.ones(input.size()[0]).to(device=input.device)
        loss = self.func(input, target, index)
        self.update_running_loss(loss)
        return loss
