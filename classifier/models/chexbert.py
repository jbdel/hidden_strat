import torch.nn as nn
import pandas as pd
from transformers import BertModel, AutoModel, AutoConfig
import torch
from collections import OrderedDict
from transformers import BertTokenizer

class bert_labeler(nn.Module):
    def __init__(self):
        """ Init the labeler module
        @param p (float): p to use for dropout in the linear heads, 0.1 by default is consistant with
                          transformers.BertForSequenceClassification
        @param clinical (boolean): True if Bio_Clinical BERT desired, False otherwise. Ignored if
                                   pretrain_path is not None
        @param freeze_embeddings (boolean): true to freeze bert embeddings during training
        @param pretrain_path (string): path to load checkpoint from
        """
        super(bert_labeler, self).__init__()
        config = AutoConfig.from_pretrained('bert-base-uncased')
        self.bert = AutoModel.from_config(config)
        self.hidden_size = self.bert.pooler.dense.in_features

    def forward(self, source_padded, attention_mask):
        """ Forward pass of the labeler
        @param source_padded (torch.LongTensor): Tensor of word indices with padding, shape (batch_size, max_len)
        @param attention_mask (torch.Tensor): Mask to avoid attention on padding tokens, shape (batch_size, max_len)
        @returns out (List[torch.Tensor])): A list of size 14 containing tensors. The first 13 have shape
                                            (batch_size, 4) and the last has shape (batch_size, 2)
        """
        # shape (batch_size, max_len, hidden_size)
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        # shape (batch_size, hidden_size)
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        return cls_hidden


class CheXbert(nn.Module):
    def __init__(self, chexbert_pth):
        super(CheXbert, self).__init__()
        model = bert_labeler()
        model_param_names = [name for name, _ in model.named_parameters()]

        state_dict = torch.load(chexbert_pth)['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            if name not in model_param_names:
                print('CheXbert: skipping param', name)
                continue
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        self.model = model
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        #     if 'layer.11' in name or 'pooler' in name:
        #         param.requires_grad = True
        # self.output = nn.Sequential(nn.Linear(self.model.hidden_size, 4096),
        #                             nn.ReLU(),
        #                             nn.Linear(4096, self.model.hidden_size))
        self.output = nn.Sequential(nn.Linear(self.model.hidden_size, self.model.hidden_size), nn.Tanh())

    def forward(self, batch, attn_mask):
        out = self.model(batch, attn_mask)
        out = self.output(out)
        return out

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # inp = "Minimal patchy airspace disease within the lingula, may reflect atelectasis or consolidation."
    # impressions = pd.Series([inp])
    # out, mask = tokenize(impressions, tokenizer)
    # print([o for o in out])
    # batch = torch.LongTensor(out).cuda()
    #
    # src_len = [b.shape[0] for b in batch]
    #
    #
    # attn_mask = generate_attention_masks(batch, src_len)
    # print(attn_mask)
    # print(mask)
    # troll
    # model = CheXbert().cuda()
    # model(batch, attn_mask)
