import torch.nn as nn
from torchvision.models import *
from .chexbert import CheXbert
from transformers import BertTokenizer


class CNN(nn.Module):
    def __init__(self, name, num_classes=None, pretrained=True, dropout=0.5, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.net = eval(name)(pretrained=pretrained)
        self.dropout = dropout
        # Overriding classifier
        if 'densenet' in name:
            self.fc_name = 'classifier'
        elif 'resnet' in name:
            self.fc_name = 'fc'
        else:
            raise NotImplementedError

        self.in_features = getattr(self.net, self.fc_name).in_features
        setattr(self.net, self.fc_name, nn.Sequential(nn.Dropout(p=self.dropout),
                                                      nn.Linear(self.in_features, self.num_classes)))

    def forward(self, sample):
        input = sample['img']
        return {'label': self.net(input.cuda())}


class CNNConstrained(CNN):
    def __init__(self, vector_size, **kwargs):
        super(CNNConstrained, self).__init__(**kwargs)
        self.vector_size = vector_size
        setattr(self.net, self.fc_name, nn.Sequential(nn.Dropout(p=self.dropout),
                                                      nn.Linear(self.in_features, vector_size)))

        self.out = nn.Linear(vector_size, self.num_classes)

    def forward(self, sample):
        vector = self.net(sample['img'].cuda())
        label = self.out(vector)
        return {'vector': vector,
                'label': label}


class CNNChexbert(CNNConstrained):
    def __init__(self, chexbert_pth, **kwargs):
        super(CNNChexbert, self).__init__(**kwargs)
        self.chexbert = CheXbert(chexbert_pth)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, sample):
        inp = self.tokenizer(sample["report"]["impression"], return_tensors="pt", padding=True)
        vector = self.net(sample['img'].cuda())
        label = self.out(vector)
        chex_vector = self.chexbert(inp.input_ids.cuda(), inp.attention_mask.cuda())
        return {'vector': vector,
                'label': label,
                'chex_vector': chex_vector,
                }
