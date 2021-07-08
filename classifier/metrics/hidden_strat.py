import copy
import json
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from dataloaders import *
from omegaconf import OmegaConf
from collections import Counter
from omegaconf.dictconfig import DictConfig


class HiddenStratMetric(nn.Module):
    def __init__(self, cfg, decision_function=None, **kwargs):
        super(HiddenStratMetric, self).__init__()
        self.cfg = cfg
        # Decision function
        assert decision_function is not None, 'decision_function is None'
        self.decision_function = decision_function
        if self.decision_function == 'softmax':
            self.pred_fn = lambda x: np.argmax(x, axis=-1)
        elif self.decision_function == 'sigmoid':
            self.pred_fn = lambda x: (x > 0)
        else:
            raise NotImplementedError(self.decision_function)

        # Get hidden_strat classes for current task (i.e. classes that have subclasses)
        self.current_task_dset = eval(self.cfg.dataset)('val', **cfg.dataset_params)
        self.class_with_subtask_name = list(self.current_task_dset.hidden_strat_label_task)
        print('Running hidden stratification on classes', self.class_with_subtask_name)
        self.tree_hidden = {label: list(self.current_task_dset.get_tree()[label]) for label in
                            self.class_with_subtask_name}
        print(json.dumps(self.tree_hidden, indent=4))

        # get position of the classes in the prediction vector
        self.class_with_subtask_pos = [self.current_task_dset.pos_label_task[label] for label in
                                       self.class_with_subtask_name]

        # Creating a dataset with task "all" to get "all" labels
        dataset_params = copy.deepcopy(cfg.dataset_params)
        OmegaConf.update(dataset_params, 'task', 'all', merge=False)
        OmegaConf.update(dataset_params, 'return_image', False, merge=False)
        all_task_val_dset = eval(self.cfg.dataset)('val', **dataset_params)

        self.y_true_all = np.array(
            [all_task_val_dset.__getitem__(i)["label"] for i in range(len(all_task_val_dset))])
        self.all_class_names = all_task_val_dset.get_all_class_names_ordered()

        # Finally, initialize counters
        self.all_class_counter = Counter()
        self.class_predicted_counter = Counter()

    def forward(self, input, target, **kwargs):
        input, target = np.array(input['label']), np.array(target['label'])
        y_pred = self.pred_fn(input)
        y_true = target

        # Small sanitiy check
        assert len(y_pred) == len(y_true) == len(self.y_true_all)

        for y_p, y_t, all_y_t in tqdm(zip(y_pred, y_true, self.y_true_all)):
            # get positive in 'all class' vector,  will be used for statistics
            all_y_t_positive_name = [self.all_class_names[index] for index in all_y_t.nonzero()[0]]
            self.all_class_counter.update(all_y_t_positive_name)

            # get positive class from pred et ground_truth
            p_pos = y_p.nonzero()[0]
            t_pos = y_t.nonzero()[0]

            # Class we care about that should be predicted
            true_class_with_subtask_pos = set.intersection(*map(set, [t_pos, self.class_with_subtask_pos]))
            class_names = [self.class_with_subtask_name[self.class_with_subtask_pos.index(pos)] for pos in
                           true_class_with_subtask_pos]
            self.all_class_counter.update(class_names)

            # if class we care about is predicted by model
            hidden_pos = set.intersection(*map(set, [p_pos, true_class_with_subtask_pos]))
            if len(hidden_pos) == 0:
                continue

            # Now we need to find the actual subclass of predicted hidden_strat class
            for pos in hidden_pos:
                class_name = self.class_with_subtask_name[self.class_with_subtask_pos.index(pos)]
                # get subclass of classname
                class_name_subclass = self.current_task_dset.get_tree()[class_name]

                # intersect with all_y_t_positive_name
                positive_sub_class_name = set.intersection(*map(set, [class_name_subclass, all_y_t_positive_name]))

                # update counter
                self.class_predicted_counter.update(list(positive_sub_class_name) + [class_name])

        stats = {k: (str(v) + '/' + str(self.all_class_counter[k])) for k, v in self.class_predicted_counter.items()}
        print(stats)
        # putting stats and self.tree_hidden all together
        return {'hidden_strat_metric': json.dumps(
            {'{} ({})'.format(k, stats[k]): ['{} ({})'.format(v_, stats[v_]) for v_ in v]
             for k, v in self.tree_hidden.items()},
            indent=4, sort_keys=True)}

    def get_required_keys(self):
        return ['label']


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
