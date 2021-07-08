from .models.cnn import *
from .losses import *
from .metrics import *


def logwrite(log, s, to_print=True):
    if to_print:
        print(s)
    log.write(str(s) + "\n")


def flatten(A):
    rt = []
    for i in A:
        if isinstance(i, list):
            rt.extend(flatten(i))
        else:
            rt.append(i)
    return rt


def get_losses_fn(cfg):
    losses_fn = []
    for loss in cfg.losses:
        loss_param = dict()
        if cfg.losses_params and loss in cfg.losses_params:
            loss_param = cfg.losses_params[loss]
        losses_fn.append(eval(loss)(cfg, **loss_param))
    return losses_fn


def get_metrics(cfg):
    if cfg.metrics_params is None:
        cfg.metrics_params = dict()
    return [eval(metric)(cfg, **cfg.metrics_params) for metric in cfg.metrics]


def get_model(cfg):
    if cfg.dataset_params.return_report:
        return CNNChexbert
    elif 'CosineLoss' in cfg.losses and ('resnet' in cfg.model or 'densenet' in cfg.model):
        return CNNConstrained
    elif 'resnet' in cfg.model or 'densenet' in cfg.model:
        return CNN
    else:
        raise NotImplementedError
