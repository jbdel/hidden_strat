import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from embeddings.utils import get_report
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizer
from transformers import BertModel, AutoModel, AutoConfig
import torch
import numpy as np


def generate_attention_masks(batch, source_lengths):
    """Generate masks for padded batches to avoid self-attention over pad tokens
    @param batch (Tensor): tensor of token indices of shape (batch_size, max_len)
                           where max_len is length of longest sequence in the batch
    @param source_lengths (List[Int]): List of actual lengths for each of the
                           sequences in the batch
    @param device (torch.device): device on which data should be

    @returns masks (Tensor): Tensor of masks of shape (batch_size, max_len)
    """
    masks = torch.ones(batch.size(0), batch.size(1), dtype=torch.float)
    for idx, src_len in enumerate(source_lengths):
        masks[idx, src_len:] = 0
    return masks.cuda()


class bert_labeler(nn.Module):
    def __init__(self, p=0.1, clinical=False, freeze_embeddings=False, pretrain_path=None, inference=False):
        """ Init the labeler module
        @param p (float): p to use for dropout in the linear heads, 0.1 by default is consistant with
                          transformers.BertForSequenceClassification
        @param clinical (boolean): True if Bio_Clinical BERT desired, False otherwise. Ignored if
                                   pretrain_path is not None
        @param freeze_embeddings (boolean): true to freeze bert embeddings during training
        @param pretrain_path (string): path to load checkpoint from
        """
        super(bert_labeler, self).__init__()

        if pretrain_path is not None:
            self.bert = BertModel.from_pretrained(pretrain_path)
        elif clinical:
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif inference:
            config = AutoConfig.from_pretrained('bert-base-uncased')
            self.bert = AutoModel.from_config(config)
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p)
        # size of the output of transformer's last layer
        hidden_size = self.bert.pooler.dense.in_features
        # classes: present, absent, unknown, blank for 12 conditions + support devices
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        # classes: yes, no for the 'no finding' observation
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

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
        # cls_hidden = self.dropout(cls_hidden)
        # out = []
        # for i in range(14):
        #     out.append(self.linear_heads[i](cls_hidden))
        return cls_hidden


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


class CheXbert(nn.Module):
    def __init__(self, cfg):
        super(CheXbert, self).__init__()
        self.cfg = cfg

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = bert_labeler(inference=True)
        checkpoint = torch.load("./embeddings/models/chexbert.pth")
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        model.cuda()
        self.model = model

    def forward(self, sample):
        inp = get_report(sample['report'], policy=self.cfg.report.report_policy)
        impressions = pd.Series([inp])
        out = tokenize(impressions, self.tokenizer)
        batch = torch.LongTensor([o for o in out])
        src_len = [b.shape[0] for b in batch]
        attn_mask = generate_attention_masks(batch, src_len)
        out = self.model(batch, attn_mask)
        return out[0].cpu().data.numpy()
