import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys
from .utils import logwrite
import collections
from omegaconf.dictconfig import DictConfig
import operator


def train(net, losses_fn, metrics_fn, train_loader, eval_loader, optimizer, scheduler, cfg):
    logfile = open(os.path.join(cfg.checkpoint_dir, 'log_run.txt'), 'w+')

    logwrite(logfile, str(cfg), to_print=False)
    logwrite(logfile, "Total number of parameters : " + str(
        sum([p.numel() for p in net.parameters() if p.requires_grad]) / 1e6) + "M")

    cfg.run = DictConfig({'no_improvements': 0,
                          'current_epoch': 0,
                          'early_stop': 0,
                          'best_early_stop_metric': 0.0 if cfg.early_stop.higher_is_better else float('inf'),
                          })

    scaler = torch.cuda.amp.GradScaler()

    summary = ''
    batch_size = cfg.hyperparameter.batch_size
    metric_comparison_func = operator.gt if cfg.early_stop.higher_is_better else operator.lt

    for epoch in range(0, 9999):
        cfg.run.current_epoch = epoch
        net.train()
        time_start = time.time()
        for step, sample in enumerate(train_loader):
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                pred = net(sample)

            losses = 0.0
            for loss_fn in losses_fn:
                loss = loss_fn.weight * loss_fn(pred, sample)
                losses += loss
            scaler.scale(losses).backward()

            # Gradient norm clipping
            if cfg.hyperparameter.grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    cfg.hyperparameter.grad_norm_clip
                )

            scaler.step(optimizer)
            scaler.update()

            summary = "\r[Epoch {}][Step {}/{}] Loss: {}, Lr: {}, ES: {}/{} ({}: {:.2f}) - {:.2f} m remaining".format(
                cfg.run.current_epoch + 1,
                step,
                int(len(train_loader.dataset) / batch_size),
                ["{}: {:.2f}".format(type(loss_fn).__name__, loss_fn.mean_running_loss) for loss_fn in losses_fn],
                *[group['lr'] for group in optimizer.param_groups],
                cfg.run.no_improvements,
                cfg.early_stop.no_improvements,
                cfg.early_stop.early_stop_metric,
                cfg.run.best_early_stop_metric,
                ((time.time() - time_start) / (step + 1)) * (
                        (len(train_loader.dataset) / batch_size) - step) / 60,
            )
            print(summary, end='          ')

        time_end = time.time()
        elapse_time = time_end - time_start
        print('Finished in {}s'.format(int(elapse_time)))
        logwrite(logfile, summary)

        if epoch + 1 >= cfg.hyperparameter.eval_start:
            metrics = evaluate(net, losses_fn, metrics_fn, eval_loader, cfg)
            logwrite(logfile, metrics)
            metric_value = metrics[cfg.early_stop.early_stop_metric]
            cfg.run.no_improvements += 1

            # Best model beaten
            if metric_comparison_func(metric_value, cfg.run.best_early_stop_metric):
                torch.save(
                    {
                        'state_dict': net.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'scheduler': scheduler.state_dict(),
                        'cfg': cfg,
                        'metrics': metrics,
                    },
                    os.path.join(cfg.checkpoint_dir, 'best.pkl')
                )
                cfg.run.no_improvements = 0
                cfg.run.best_early_stop_metric = float(metric_value)

            # Scheduler
            if cfg.scheduler.use_scheduler:
                scheduler.step(metrics[cfg.early_stop.early_stop_metric])

        # Early stop ?
        if cfg.run.no_improvements == cfg.early_stop.no_improvements:
            import sys
            os.rename(os.path.join(cfg.checkpoint_dir, 'best.pkl'),
                      os.path.join(cfg.checkpoint_dir, 'best' + str(cfg.run.best_early_stop_metric) + '.pkl'))
            print('Early stop reached')
            sys.exit()


def evaluate(net, losses_fn, metrics_fn, eval_loader, cfg):
    print('Evaluation...')
    net.eval()
    with torch.no_grad():
        preds = collections.defaultdict(list)
        samples = collections.defaultdict(list)

        # Getting all prediction and labels
        for step, sample in (enumerate(eval_loader)):
            pred = net(sample)
            for k in pred.keys():
                preds[k].extend(pred[k].cpu().data.numpy())
            for k in sample.keys():
                sample[k] = sample[k].data.numpy() if isinstance(sample[k], torch.Tensor) else sample[k]
                samples[k].extend(sample[k])

        metrics = dict()
        # calculating validation loss(es)
        for loss_fn in losses_fn:
            metrics[type(loss_fn).__name__] = loss_fn(preds, samples).item()
        # calculating metrics
        for metric_fn in metrics_fn:
            metrics = {**metrics, **metric_fn(preds, samples, net=net)}

    return metrics
