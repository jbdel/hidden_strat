import torch.nn as nn
from embeddings.utils import get_report
from gensim.models import Doc2Vec
from tqdm import tqdm
import os
import gensim
import multiprocessing
from gensim.models.callbacks import CallbackAny2Vec
from dataloaders import *


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        print("\rEpoch #{} end".format(self.epoch),
              end='          ')
        self.epoch += 1

    def on_training_end(self, model):
        print('\n')


def train_doc2vec(cfg):
    doc2vec = gensim.models.doc2vec.Doc2Vec(dm=0,
                                            dbow_words=0,
                                            vector_size=cfg.model.vector_size,
                                            window=8,
                                            min_count=15,
                                            epochs=cfg.model.epochs,
                                            workers=multiprocessing.cpu_count(),
                                            callbacks=[EpochLogger()])

    # Build corpus
    train_corpus = []
    dataset = eval(cfg.dataset.name)('train',
                                     return_report=True,
                                     return_label=False,
                                     return_image=False)
    print("Building corpus")
    for i, sample in enumerate(tqdm(dataset)):
        report = get_report(sample['report'], policy=cfg.report.report_policy)
        report = gensim.utils.simple_preprocess(report)
        train_corpus.append(gensim.models.doc2vec.TaggedDocument(report, [i]))

    # Build vocab
    doc2vec.build_vocab(train_corpus)
    print("Corpus contains " + str(len(train_corpus)) + " reports \n" +
          "Vocabulary count : " + str(len(doc2vec.wv.vocab)) + ' words \n' +
          "Corpus total words : " + str(doc2vec.corpus_total_words) + " words \n" +
          "Corpus count : " + str(doc2vec.corpus_count))

    # Train the model
    print("Training model")
    doc2vec.train(train_corpus, total_examples=doc2vec.corpus_count, epochs=doc2vec.epochs)

    # Save the model
    checkpoint = os.path.join(cfg.outdir, 'DBOW_vector' +
                              str(doc2vec.vector_size) +
                              '_window' +
                              str(doc2vec.window) +
                              '_count' +
                              str(doc2vec.vocabulary.min_count) +
                              '_epoch' +
                              str(doc2vec.epochs) +
                              '_mimic.doc2vec')

    doc2vec.save(checkpoint)
    print("Model saved")
    return checkpoint


class Doc2vec(nn.Module):
    def __init__(self, cfg):
        super(Doc2vec, self).__init__()
        self.cfg = cfg
        if cfg.model.to_train:
            checkpoint = train_doc2vec(cfg)
        else:
            checkpoint = cfg.model.checkpoint

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)
        self.model = Doc2Vec.load(checkpoint)

    def forward(self, sample):
        inp = get_report(sample['report'], policy=self.cfg.report.report_policy)
        inp = gensim.utils.simple_preprocess(inp)
        return self.model.infer_vector(inp)
