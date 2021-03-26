import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from embeddings.utils import get_report


class BioClinicalBERT(nn.Module):
    def __init__(self, cfg):
        super(BioClinicalBERT, self).__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", return_dict=True)

    def forward(self, sample):
        inp = get_report(sample['report'], policy=self.cfg.report.report_policy)
        inp = self.tokenizer(inp, return_tensors="pt")
        return self.model(**inp).pooler_output.cpu().data.numpy().squeeze(0)
