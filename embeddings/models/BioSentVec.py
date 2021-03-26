import torch.nn as nn
from embeddings.utils import get_report, download_ressource

try:
    import sent2vec
except ModuleNotFoundError:
    raise ModuleNotFoundError("Install sent2vec from https://github.com/epfml/sent2vec")

import os
import nltk
from nltk import sent_tokenize, word_tokenize

nltk.download('punkt')


class BioSentVec(nn.Module):
    def __init__(self, cfg):
        super(BioSentVec, self).__init__()
        self.cfg = cfg
        checkpoint = cfg.model.checkpoint
        if not os.path.exists(checkpoint):
            if cfg.model.checkpoint_download:
                download_ressource(checkpoint,
                                   'https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin')
            else:
                raise FileNotFoundError(checkpoint)
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(checkpoint)

    def forward(self, sample):
        inp = get_report(sample['report'], policy=self.cfg.report.report_policy)
        inp = ' '.join(word_tokenize(inp))
        return self.model.embed_sentence(inp).squeeze(0)
