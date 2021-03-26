import torch.nn as nn
import os
import torch
from classifier.models import *
import numpy as np


class CNNVec(nn.Module):
    def __init__(self, cfg):
        super(CNNVec, self).__init__()
        self.cfg = cfg
        checkpoint = cfg.model.checkpoint
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)

        ckpt = torch.load(checkpoint)
        ckpt_cfg = ckpt['cfg']
        model_name = ckpt_cfg.model_params.net_func
        model = eval(model_name)(**ckpt_cfg.model_params)

        if model_name == 'CNN':
            self.fc = getattr(model.net, model.fc_name)
        elif model_name == 'CNNConstrained':
            self.fc = getattr(model, 'out')
        else:
            raise NotImplementedError(model_name)

        self.net = nn.DataParallel(model)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.cuda()
        self.net.eval()

        self.hook = None
        def hook(m, i, o):
            self.hook = i[0].squeeze().data
        self.fc.register_forward_hook(hook)

    def forward(self, sample):
        with torch.no_grad():
            sample['img'] = sample['img'].unsqueeze(0)
            _ = self.net(sample)
            return self.hook.cpu().numpy()
