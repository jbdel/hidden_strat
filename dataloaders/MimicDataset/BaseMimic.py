import numpy as np
from dataloaders.BaseDataset import BaseDataset


class BaseMimic(BaseDataset):
    def __init__(self, task='all', data_root='./data/mimic-cxr/', image_root='./data/mimic-cxr/images',
                 ann_file='annotations.json', **kwargs):
        super().__init__(task)
        self.data_root = data_root
        self.image_root = image_root
        self.ann_file = ann_file

    @staticmethod
    def get_all_class_names_ordered():
        return ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                'Pneumothorax', 'Support Devices']

    def get_tasks(self):
        return ['all', 'binary', 'six']

    def get_tree(self):
        if self.task == 'binary':
            return {'No Finding': {'No Finding'},
                    'Findings': {'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                                 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
                                 'Pneumonia', 'Pneumothorax', 'Support Devices'}}

        elif self.task == 'six':
            # according to https://stanfordmlgroup.github.io/competitions/chexpert/img/figure1.png
            return {'No Finding': {'No Finding'},
                    'Support Devices': {'Support Devices'},
                    'Fracture': {'Fracture'},
                    'Lung': {'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Lung Lesion',
                             'Atelectasis'},
                    'Cardio': {'Enlarged Cardiomediastinum', 'Cardiomegaly'},
                    'Pleural': {'Pleural Other', 'Pleural Effusion', 'Pneumothorax'}
                    }

        elif self.task == 'all':
            return {v: [v] for v in BaseMimic.get_all_class_names_ordered()}

    def get_encoded_label(self, label):
        """
        Dataset specific label processing
        :param label: 1-D python list
        """
        label = np.array(label).astype(np.float)
        label[label < 0] = 0.0
        # print(self.print_label(label, self.get_all_class_names_ordered()))

        # If no label is specified, put No Finding
        if sum(label) == 0:
            label[self.pos_label_all['No Finding']] = 1.0

        # When task is binary we cant have No findings and Support Devices to coexist
        # This would return the label [1,1] according to the binary tree.
        if self.task == 'binary' \
                and (label[self.pos_label_all['Support Devices']] == 1.0) \
                and (label[self.pos_label_all['No Finding']] == 1.0):
            label[self.pos_label_all['Support Devices']] = 0.0
        return super().get_encoded_label(label)
