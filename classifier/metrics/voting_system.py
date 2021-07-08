import numpy as np
from dataloaders import *
from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf
from sklearn.utils.sparsefuncs import count_nonzero
from scipy.sparse import csr_matrix
import copy
from sklearn.metrics import classification_report
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler

def get_labels(dset):
    labels = []
    for i in tqdm(range(len(dset))):
        sample = dset.__getitem__(i)
        labels.append(sample["label"])
    return np.array(labels)


def get_vectors_dataloader(dset):
    vectors = []
    for i in tqdm(range(len(dset))):
        # for i in tqdm(range(500)):
        sample = dset.__getitem__(i)
        vectors.append(sample["vector"])
    return np.array(vectors)


def get_vectors_model(dset, net):
    vectors = []
    dset.return_image = True
    net.eval()
    bs = BatchSampler(SequentialSampler(dset), batch_size=1, drop_last=False)
    dl = DataLoader(dset, batch_sampler=bs)
    for s in tqdm(dl):
        pred = net(s)
        vectors.append(pred["vector"].squeeze().cpu().data.numpy())
    return np.array(vectors)


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


# https://github.com/RaRe-Technologies/gensim/blob/master/gensim/matutils.py#L48
def argsort(x, topn=None, reverse=False):
    """Efficiently calculate indices of the `topn` smallest elements in array `x`."""
    x = np.asarray(x)  # unify code path for when `x` is not a np array (list, tuple...)
    if topn is None:
        topn = x.size
    if topn <= 0:
        return []
    if reverse:
        x = -x
    if topn >= x.size or not hasattr(np, 'argpartition'):
        return np.argsort(x)[:topn]
    # np >= 1.8 has a fast partial argsort, use that!
    most_extreme = np.argpartition(x, topn)[:topn]
    return most_extreme.take(np.argsort(x.take(most_extreme)))  # resort topn into order


class VotingSystemMetric(nn.Module):
    def __init__(self, cfg, decision_function=None, vectors_from_model=False, **kwargs):
        super(VotingSystemMetric, self).__init__()

        self.cfg = cfg
        # whether vectors comes from dataloader or model (True if finetune chexbert for e.g.)
        self.vectors_from_model = vectors_from_model

        # Decision function
        assert decision_function is not None, 'decision_function is None'
        self.decision_function = decision_function
        if self.decision_function == 'softmax':
            self.pred_fn = lambda x: np.argmax(x, axis=-1)
        elif self.decision_function == 'sigmoid':
            self.pred_fn = lambda x: (x > 0)
        else:
            raise NotImplementedError(self.decision_function)

        # We need the labels from "all" task to compute metrics
        dataset_params = copy.deepcopy(cfg.dataset_params)
        OmegaConf.update(dataset_params, 'task', 'all', merge=False)
        OmegaConf.update(dataset_params, 'return_image', False, merge=False)
        train_dset = eval(self.cfg.dataset)('train', **dataset_params)
        val_dset = eval(self.cfg.dataset)('val', **dataset_params)

        print('VotingSystemMetric: Building list of labels')
        self.train_labels = get_labels(train_dset)
        self.valid_labels = get_labels(val_dset)

        # Getting the training linguistic embedding space (if from dataloader)
        if not self.vectors_from_model:
            print('VotingSystemMetric: Building list of vectors')
            self.train_labels = get_vectors_dataloader(train_dset)

        # Get 'all' info
        self.all_classes = np.array(val_dset.get_all_class_names_ordered())
        self.pos_label_all = val_dset.pos_label_all

        # Get 'task' info
        val_dset.task = cfg.dataset_params.task
        self.task_tree = val_dset.get_tree()
        self.task_classes = np.array(list(val_dset.get_classes()))

        self.train_dset = train_dset
        self.val_dset = val_dset

    def forward(self, input, target, net=None, **kwargs):

        # CNN last states from validation set
        vectors = torch.from_numpy(np.array(input['vector']))

        # if the training embedding space has been finetuned, we compute it now
        if self.vectors_from_model:
            assert net is not None
            print('VotingSystemMetric: Building list of vectors')
            self.train_vectors = get_vectors_model(self.train_dset, net)

        dists = cosine_distance_torch(vectors.cuda(), torch.from_numpy(self.train_vectors).cuda())
        dists = dists.cpu().data.numpy()

        # Getting preds
        y_pred = self.pred_fn(np.array(input['label']))
        # y_true = self.pred_fn(np.array(target['label']))
        # correct = np.where(count_nonzero(csr_matrix(y_true) - csr_matrix(y_pred), axis=1) == 0)
        # correct = correct[0]

        y_true_all = []
        y_pred_all = []
        for index in tqdm(range(len(y_pred))):
            # Get top 10 train neighbors
            best_index = argsort(dists[index], topn=10)
            # get neighbors labels and average it, we keep labels that appears > 50% of the time
            neighbors_labels = self.train_labels[best_index]
            neighbors_positive = (np.mean(neighbors_labels, axis=0)) > 0.5

            # get neighbors classes according to 'all' task
            report_classes = np.array(self.all_classes)[neighbors_positive]

            # get task class predicted by model
            pred = np.squeeze(y_pred[index])
            c_name = self.task_classes[np.where(pred == 1)]

            # Get 'all' ground truth label
            gt_all = np.squeeze(self.valid_labels[index])
            pred_all = np.zeros(len(gt_all))

            # For each predicted class
            for c in c_name:
                # Get subclasses
                subclasses = self.task_tree[c]
                # If no subclasses, we put the label where it belongs
                if len(subclasses) == 1:
                    pred_all[self.pos_label_all[c]] = 1.
                    continue
                # for each class found in report, if its a subclass, put the label where it belongs
                for r_c in report_classes:
                    if r_c in subclasses:
                        pred_all[self.pos_label_all[r_c]] = 1.

            y_pred_all.append(pred_all)
            y_true_all.append(gt_all)

        metrics = dict()

        metrics['hidden_strat_report_string'] = classification_report(np.array(y_true_all), np.array(y_pred_all),
                                                                      target_names=list(self.all_classes))
        metrics['hidden_strat_report_dict'] = classification_report(np.array(y_true_all), np.array(y_pred_all),
                                                                    target_names=list(self.all_classes),
                                                                    output_dict=True)
        return metrics

    def get_required_keys(self):
        return ['label', 'vector']


def valid(o):
    o(input={'label': [[0, 1], [1, 0]]},
      target={'label': [[0, 1], [0, 1]],
              'vector': np.zeros((2, 768))})


if __name__ == '__main__':
    cfg = DictConfig({'model': 'densenet169',
                      'dataset': 'VectorMimicDataset',
                      'dataset_params': {'task': 'six', 'return_label': True, 'return_image': True,
                                         'return_report': False,
                                         'vector_file': 'linguistics/embeddings/output/doc2vec_mimic_mit/vectors.pkl',
                                         'num_classes': 6,
                                         'task_classes': ['No Finding', 'Support Devices', 'Fracture', 'Lung Opacity',
                                                          'Enlarged Cardiomediastinum', 'Pleural']},
                      })

    o = VotingSystemMetric(cfg=cfg, decision_function='sigmoid')
    valid(o)
