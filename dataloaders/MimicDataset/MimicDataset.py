import os
import json
import torch
from PIL import Image
from .BaseMimic import BaseMimic
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader


class MimicDataset(BaseMimic):
    def __init__(self, split, return_image=False, return_label=False, return_report=False, **kwargs):
        super(MimicDataset, self).__init__(**kwargs)
        assert split in ['train', 'val', 'test']
        self.split = split

        self.return_image = return_image
        self.return_report = return_report
        self.return_label = return_label
        self.transform = MimicDataset.get_transforms(split)

        self.ann = json.loads(open(os.path.join(self.data_root, self.ann_file), 'r').read())
        self.examples = self.ann[split]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        study_id = example['study_id']
        subject_id = example['subject_id']

        key = (subject_id, study_id)

        image = torch.tensor(0)
        label = torch.tensor(0)
        report = torch.tensor(0)

        if self.return_image:
            image_path = example['image_path']
            try:
                image = self.transform(Image.open(os.path.join(self.image_root, image_path)).convert('RGB'))
            except FileNotFoundError:
                print('image not found for key', image_path)
                raise

        if self.return_report:
            report = example['report']

        if self.return_label:
            label = self.get_encoded_label(example['label'])

        return {'idx': idx,
                'key': key,
                'report': report,
                'img': image,
                'label': label}

    @staticmethod
    def get_transforms(name):
        if name == 'train':
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])


if __name__ == '__main__':
    d = MimicDataset("train",
                     return_image=True,
                     return_label=True,
                     return_report=True,
                     task='six')
    l = DataLoader(d,
                   batch_size=2)
    print(len(d))
    for s in tqdm(l):
        continue
