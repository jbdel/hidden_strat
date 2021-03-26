import os
import pickle
import numpy as np
from .basemetric import BaseMetric
from dataloaders import *
from tqdm import tqdm
from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf
from sklearn.utils.sparsefuncs import count_nonzero
from scipy.sparse import csr_matrix
import copy
from collections import defaultdict, Counter
from sklearn.metrics import classification_report


def get_vectors_labels(dset):
    vectors = []
    labels = []
    for i in tqdm(range(len(dset))):
    # for i in tqdm(range(500)):
        sample = dset.__getitem__(i)
        vectors.append(sample["vector"])
        labels.append(sample["label"])
    return np.array(vectors), np.array(labels)


def argsort(x, topn=None, reverse=False):
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


class HiddenStratMetric(BaseMetric):
    def __init__(self, cfg, decision_function=None, **kwargs):
        super(HiddenStratMetric, self).__init__(cfg, **kwargs)

        # Decision function
        assert decision_function is not None, 'decision_function is None'
        self.decision_function = decision_function
        if self.decision_function == 'softmax':
            self.pred_fn = lambda x: np.argmax(x, axis=-1)
        elif self.decision_function == 'sigmoid':
            self.pred_fn = lambda x: (x > 0)
        else:
            raise NotImplementedError(self.decision_function)

        # Train vectors and train labels for 'all' task
        dataset_params = copy.deepcopy(cfg.dataset_params)
        OmegaConf.update(dataset_params, 'task', 'all', merge=False)
        train_dset = eval(self.cfg.dataset)('train', **dataset_params)
        val_dset = eval(self.cfg.dataset)('val', **dataset_params)

        report_pkl_path = os.path.join(train_dset.data_root, "hidden_strat.pkl")
        if not os.path.exists(report_pkl_path):
            print('HiddenStratMetric: Building list of reports (only once)')
            train_vectors, train_labels = get_vectors_labels(train_dset)
            valid_vectors, valid_labels = get_vectors_labels(val_dset)
            pickle.dump((train_vectors, train_labels, valid_vectors, valid_labels),
                        open(report_pkl_path, "wb"))

        self.train_vectors, self.train_labels, self.valid_vectors, self.valid_labels = pickle.load(
            open(report_pkl_path, "rb"))

        # Get 'all' info
        self.all_classes = np.array(val_dset.get_all_class_names_ordered())
        self.pos_label_all = val_dset.pos_label_all

        # Get 'task' info
        val_dset.task = cfg.dataset_params.task
        self.task_tree = val_dset.get_tree()
        self.task_classes = np.array(list(val_dset.get_classes()))

        del train_dset
        del val_dset

    def forward(self, input, target):
        # import inspect        #
        # for v in inspect.stack():
        #     print(v.function)
        # sys.exit()

        input, target = super().forward(input, target)
        metrics = dict()

        # Getting correct classification
        y_pred = self.pred_fn(input['label'])
        y_true = self.pred_fn(target['label'])
        correct = np.where(count_nonzero(csr_matrix(y_true) - csr_matrix(y_pred), axis=1) == 0)
        correct = correct[0]

        y_true_all = []
        y_pred_all = []
        # For each right prediction...
        for index in correct:
            # Get top 10 train neighbors
            vector = target['vector'][index]
            dists = np.dot(self.train_vectors, np.squeeze(vector))
            best_index = argsort(dists, topn=10, reverse=True)
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

        metrics['hidden_strat_report_string'] = classification_report(np.array(y_true_all), np.array(y_pred_all),
                                                                      target_names=self.all_classes)
        metrics['hidden_strat_report_dict'] = classification_report(np.array(y_true_all), np.array(y_pred_all),
                                                                    target_names=self.all_classes,
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

    o = HiddenStratMetric(cfg=cfg, decision_function='sigmoid')
    valid(o)
