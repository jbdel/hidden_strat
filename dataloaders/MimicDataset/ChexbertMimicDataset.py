from __future__ import print_function
from .MimicDataset import MimicDataset
from tqdm import tqdm
from transformers import BertTokenizer
import pandas as pd


def tokenize(impressions, tokenizer):
    imp = impressions.str.strip()
    imp = imp.replace('\n', ' ', regex=True)
    imp = imp.replace('\s+', ' ', regex=True)
    impressions = imp.str.strip()
    new_impressions = []
    for i in (range(impressions.shape[0])):
        tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
        if tokenized_imp:  # not an empty report
            res = tokenizer.encode_plus(tokenized_imp)['input_ids']
            if len(res) > 512:  # length exceeds maximum size
                # print("report length bigger than 512")
                res = res[:511] + [tokenizer.sep_token_id]
            new_impressions.append(res)
        else:  # an empty report
            new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
    return new_impressions


class ChexbertMimicDataset(MimicDataset):
    def __init__(self, split, **kwargs):
        super(ChexbertMimicDataset, self).__init__(split, **kwargs)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        report = sample["report"]
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        impressions = pd.Series([report])
        out = tokenize(impressions, tokenizer)

        key = sample['key']
        try:
            vector = self.vectors[key]
        except KeyError:
            raise KeyError(key)

        sample['vector'] = vector
        return sample


if __name__ == '__main__':
    d = ChexbertMimicDataset("test",
                             return_image=True,
                             return_label=True,
                             return_report=False)
    for _ in tqdm(d):
        continue
