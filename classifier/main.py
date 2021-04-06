import argparse, os, sys
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from dataloaders import *
from torch.optim.lr_scheduler import *
import torch.optim as optim
from torch.utils.data import DataLoader

from .train import train, evaluate
from .utils import get_losses_fn, get_model, get_metrics, flatten

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('-o', help="Overriding arguments", nargs='+', default=[])

    args = parser.parse_args()

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        config = ckpt['cfg']
    else:
        config = OmegaConf.load(args.config)

    override = OmegaConf.from_dotlist(args.o)
    if override:
        print('Overriding dict:', override)

    cfg = OmegaConf.merge(config, override)

    torch.backends.cudnn.benchmark = True

    # DataLoader
    train_dset: BaseDataset = eval(cfg.dataset)('train', **cfg.dataset_params)
    eval_dset: BaseDataset = eval(cfg.dataset)('val', **cfg.dataset_params)
    test_dset: BaseDataset = eval(cfg.dataset)('test', **cfg.dataset_params)
    cfg.dataset_params.num_classes = train_dset.num_classes
    cfg.dataset_params.task_classes = train_dset.task_classes

    n_gpus = torch.cuda.device_count()
    train_loader = DataLoader(train_dset,
                              cfg.hyperparameter.batch_size,
                              shuffle=True,
                              num_workers=n_gpus * 4,
                              pin_memory=True,
                              drop_last=True)

    eval_loader = DataLoader(eval_dset,
                             int(cfg.hyperparameter.batch_size / 2),
                             shuffle=False,
                             num_workers=n_gpus * 4,
                             pin_memory=True)
    test_loader = None
    # test_loader = DataLoader(test_dset,
    #                          int(cfg.hyperparameter.batch_size / 2),
    #                          num_workers=n_gpus * 4,
    #                          pin_memory=True)

    dataset_keys = train_dset.__getitem__(0).keys()
    print('Using dataset', type(train_dset).__name__, 'with task', train_dset.task, 'returning batch with keys',
          dataset_keys)
    # Net
    net_func = get_model(cfg)
    cfg.model_params = OmegaConf.merge({
        'name': cfg.model,
        'num_classes': train_dset.num_classes,
        'net_func': net_func.__name__
    }, cfg.model_params)

    net = net_func(**cfg.model_params).cuda()
    print('Using model', type(net).__name__, 'with input', net.get_forward_input_keys(), 'returning',
          net.get_forward_output_keys(), 'with', net.num_classes,
          'output neurons')

    # Losses
    losses_fn = get_losses_fn(cfg)
    loss_required_keys = set(flatten([loss.get_required_keys() for loss in losses_fn]))
    print('Using losses', [(type(loss).__name__, 'weight:' + str(loss.weight)) for loss in losses_fn],
          "needing the model to return",
          loss_required_keys)

    # Metrics
    metrics_fn = get_metrics(cfg)
    metrics_required_keys = set(flatten([metric.get_required_keys() for metric in metrics_fn]))
    print('Using metrics', [type(metric).__name__ for metric in metrics_fn], "needing the model to return",
          metrics_required_keys)

    # Checking configs OK
    assert all([key in dataset_keys for key in net.get_forward_input_keys()]), 'Dataset and Model keys dont match'
    assert all([key in net.get_forward_output_keys() for key in loss_required_keys]), 'Losses and Model keys dont match'
    assert all(
        [key in net.get_forward_output_keys() for key in metrics_required_keys]), 'Metrics and Model keys dont match'
    print('Everything matches')

    # Create Checkpoint dir
    cfg.checkpoint_dir = os.path.join(cfg.experiment.output_dir, cfg.experiment.name)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.hyperparameter.lr_base)
    print('Using optimizer', optimizer)

    scheduler = None
    if cfg.scheduler.use_scheduler:
        scheduler = eval(cfg.scheduler.name)(optimizer, **cfg.scheduler_params)
        print('Using scheduler', scheduler)

    net = nn.DataParallel(net)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.ckpt is None:
        # Run training
        eval_accuracies = train(net,
                                losses_fn,
                                metrics_fn,
                                train_loader,
                                eval_loader,
                                optimizer,
                                scheduler,
                                cfg,
                                )
    else:
        net.load_state_dict(ckpt['state_dict'])
        metrics = evaluate(net, losses_fn, metrics_fn, eval_loader, cfg)
        for k, v in metrics.items():
            if 'dict' in k:
                continue
            print("'{}':{}".format(k, v))

    # # Keeping all objects in cfg
    # cfg.objects = DictConfig({'train_dset': train_dset,
    #                           'eval_dset': eval_dset,
    #                           'test_dset': test_dset,
    #                           'net': net,
    #                           'losses_fn': losses_fn,
    #                           'metrics_fn': metrics_fn,
    #                           'optimizer': optimizer,
    #                           'scheduler': scheduler,
    #                           })
