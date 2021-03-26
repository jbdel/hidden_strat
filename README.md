##### tackling the hidden strat problem

Datasets are configured with different tasks. 

For example, we define mimic-cxr with three tasks: all (cnn is trained on all fourteen classes), six (according to this 
[tree](https://stanfordmlgroup.github.io/competitions/chexpert/img/figure1.png)) or binary. This definition is available in 
`dataloaders/MimicDataset/BaseMimic.py`

To run a model on MimicDataset using a densenet backbone, use the following command
```
python -m classifier.main --config classifier/configs/cnn.yml
```
By default, this command will start training on task 'six' as stated in the yml file. You can override config parameters by using `--o`:
```
python -m classifier.main --config classifier/configs/cnn.yml -o dataset_params.task=all hyperparameter.lr_base=1e-3
```

To evaluate a checkpoint, use the `ckpt` argument. According config is loaded from the ckpt.
```
python -m classifier.main --ckpt classifier/checkpoints/my_model_all/best0.46879885719564507.pkl -o metrics=[ClassificationMetric]
```

##### How to constrain a model

For a dataset, here is what annotations.json looks like:

```
{"val":
     [
        {'id': str,
        'study_id': str,
        'subject_id': str,
        'image_path': str,
        'split': str,
        'label': list,
        'report': dict,
        },
     ...],
 "train": [{...}, ...],
 "test": [{...}, ...]
 }
```
The report dictionary contains the following information:
```
{
 'findings': str,
 'impression': str,
 'last_paragraph': str,
 'comparison': str
}
```
You can use each report to compute vectors representation. This is the command using doc2vec

```
python -m linguistics.embeddings.compute_embeddings --config doc2vec_train.yml
```
It trains a doc2vec model and plots embeddings using tsne and umap. 

It will also create the file `linguistics/embeddings/output/doc2vec_mimic/vectors.pkl` that you can use to train a contrained model.<br/>

In the config file, notice the param `report.report_policy: top_section`. It defines what to input from the report to the embedding model.
So far we have two policies definied at: `embeddings/utils.py`:
 
 ```
def get_report(report, policy=None):
    if policy is None:
        policy = 'top_section'

    if policy == 'top_section':
        for section in ['findings', 'impression', 'background']:
            if section not in report:
                continue

            if report[section] != '':
                return report[section]

    elif policy == 'all_section':
        ret = ''
        for section in ['findings', 'impression', 'background']:
            if section not in report:
                continue
            ret += report[section] + ' '
    else:
        raise NotImplementedError(policy)
```

Make your own.<br/>

To train a constrained model, use the following command
```
python -m classifier.main --config classifier/configs/cnn_constrained.yml -o dataset_params.vector_file: embeddings/output/doc2vec_mimic/vectors.pkl
```
The constraining is done by the `CosineLoss` in `classifier/losses/cosine.py`.

Though it will be trained on `task: six`, you can evaluate it on all classes by doing:
```
python -m classifier.main --ckpt classifier/checkpoints/my_model_constrained/best0.46879885719564507.pkl -o metrics=[HiddenStratMetric]
```
More info on how we do it in `classifier/metrics/hidden_strat.py`
