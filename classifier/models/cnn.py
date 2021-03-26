import torch.nn as nn
from torchvision.models import *
from .basemodel import BaseModel
import torch


class CNN(BaseModel):
    def __init__(self, name, pretrained=True, **kwargs):
        super(CNN, self).__init__(**kwargs)
        self.net = eval(name)(pretrained=pretrained)

        # Overriding classifier
        if 'densenet' in name:
            self.fc_name = 'classifier'
        elif 'resnet' in name:
            self.fc_name = 'fc'
        else:
            raise NotImplementedError

        self.in_features = getattr(self.net, self.fc_name).in_features
        setattr(self.net, self.fc_name, nn.Linear(self.in_features, self.num_classes))

    def forward(self, sample):
        input = sample['img']
        return {'label': self.net(input.cuda())}

    def get_forward_output_keys(self):
        output = self.forward({'img': torch.zeros(1, 3, 224, 224)})
        return output.keys()

    def get_forward_input_keys(self):
        return ['img']


class CNNConstrained(CNN):
    def __init__(self, vector_size, **kwargs):
        super(CNNConstrained, self).__init__(**kwargs)
        setattr(self.net, self.fc_name, nn.Linear(self.in_features, vector_size))
        self.out = nn.Linear(vector_size, self.num_classes)

    def forward(self, sample):
        vector = self.net(sample['img'].cuda())
        label = self.out(vector)
        return {'vector': vector,
                'label': label}
