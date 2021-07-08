mimic-data: Download annotations.json and the images from [here](https://drive.google.com/drive/folders/1dQVcrU3NLWwYDCtsnAEI_ROh1cBwFSHV)
and place it in `data/mimic-cxr/`<br/>


##### Lets go

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
python -m classifier.main --config classifier/configs/cnn.yml -o dataset_params.task=all
```

To evaluate a checkpoint, use the `ckpt` argument. According config is loaded from the ckpt.
```
python -m classifier.main --ckpt classifier/checkpoints/my_model_all/best0.46879885719564507.pkl -o metrics=[ClassificationMetric]
```

##### How to constrain a model
1) Either you specify a pkl vector file trained with the `embeddings` package.
```
python -m classifier.main --config classifier/configs/cnn_constrained.yml -o dataset_params.vector_file=embeddings/output/doc2vec_mimic/vectors.pkl
```
This command will use the `VectorMimicDataset` dataloader.<br/>
The constraining is done by the `CosineLoss` in `classifier/losses/cosine.py`.<br/>
The reports embedding vectors are fixed and not finetuned.

Though it will be trained on `task: six`, you can evaluate it on all classes using the VotingSystemMetric by doing:
```
python -m classifier.main --ckpt classifier/checkpoints/my_model_constrained/best0.46879885719564507.pkl -o metrics=[VotingSystemMetric]
```
The voting system is as described in the ppt.

2) Or you use chexbert to compute the report embedding vectors, and finetune the model (while training the CNN). <br/>
First, download the chexbert.pth pretrained model [here](https://drive.google.com/drive/folders/17LbZabgnvQfutRnLTRIO_MsCVaFUdhjj?usp=sharing) 
and place it in `classifier/models`
```
python -m classifier.main --config classifier/configs/cnn_constrained_chexbert.yml \
-o dataset_params.task=six 
```
All infos are in the yml file. <br/>
Evaluation in this case is run with cmd :
``` 
python -m classifier.main --ckpt classifier/checkpoints/my_model_constrained/best0.07790679484605789.pkl \
-o metrics=[VotingSystemMetric] \
metrics_params.vectors_from_model=True
```