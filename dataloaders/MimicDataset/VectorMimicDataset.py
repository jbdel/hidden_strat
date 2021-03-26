from __future__ import print_function
from .MimicDataset import MimicDataset
from tqdm import tqdm
import pickle


class VectorMimicDataset(MimicDataset):
    def __init__(self, split, vector_file=None, **kwargs):
        super(VectorMimicDataset, self).__init__(split, **kwargs)
        assert vector_file is not None
        self.vectors = pickle.load(open(vector_file, 'rb'))[split]

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        key = sample['key']
        try:
            vector = self.vectors[key]
        except KeyError:
            raise KeyError(key)

        sample['vector'] = vector
        return sample


if __name__ == '__main__':
    d = VectorMimicDataset("test", "vector_file",
                           return_image=True,
                           return_label=True,
                           return_report=True)
    for _ in tqdm(d):
        continue
