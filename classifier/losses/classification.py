import torch.nn as nn
from .baseloss import BaseLoss


class ClassificationLoss(BaseLoss):
    def __init__(self, cfg, **kwargs):
        super(ClassificationLoss, self).__init__(cfg, **kwargs)
        self.func = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, input, target):
        input, target = super().forward(input, target)
        input, target = input['label'], target['label']
        return self.func(input, target)

    def get_required_keys(self):
        return ['label']
