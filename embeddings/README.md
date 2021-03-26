### Linguistic embeddings
Requirements
```
pip install pandas, sklearn, matplotlib, tqdm, transformers, nltk, gensim
```

To install sent2vec

```
git clone https://github.com/epfml/sent2vec
cd sent2vec
pip install Cython
pip install .
```

A config file should look more or less like this:
```
model:
    name: BioClinicalBERT
dataset:
    name: MimicDataset
    task: six
report:
    report_policy: top_section
experiment:
    name: bioclinicalbert_mimic_six
    save_vectors: False
    output_dir: linguistics/embeddings/output/
 ```   

The report_policy setting defines what to do with a report. It is define in the function `def get_report(report, policy=None)` in 
`utils.py`. 

For example, a report sample from MimicDataset has keys `dict_keys(['findings', 'impression', 'background', 'r2gen'])`.

The report_policy `top_section` is defined as:
```
for section in ['findings', 'impression', 'background']:
    if report[section] != '':
        return report[section]
```
 

This package computes the embeddings of reports. We evaluate [BioclinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT/), 
[BlueBert](https://github.com/ncbi-nlp/bluebert/), [BioSentVec](https://github.com/ncbi-nlp/BioSentVec) and Doc2Vec.

<b>BioClinicalBERT</b> and <b>BlueBert</b> are directly available through the [HuggingFace transformers](https://github.com/huggingface/transformers) library.
```
python -m linguistics.embeddings.compute_embeddings --config bioclinicalbert
```

<b>BioSentVec</b>
```
python -m linguistics.embeddings.compute_embeddings --config biosentvec
```
Config downloads the pretrained model if checkpoint not found (21 Go)

<b>Doc2Vec</b>
```
python -m linguistics.embeddings.compute_embeddings --config doc2vec_train
```
Trains a doc2vec model and evaluates it.

<b>CNN</b>

To get the vectors of a model trained in the classifier package, use:
```
python -m linguistics.embeddings.compute_embeddings --config CNN

```
with the right checkpoint path.
