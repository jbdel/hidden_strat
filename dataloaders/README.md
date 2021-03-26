### Dataloaders

A dataset should return a dictionary. The dictionnary must contain at least the key "key" that 
defines the unique identifier of the sample. 

The rest is up to the dataset data.

<b>```MimicDataset.py```</b> <br/>
[Download annotations.json](https://drive.google.com/drive/folders/1pU97NrwdqG9raBm4aXx4gep2FfUFE_Rp?usp=sharing)

Loads `annotations.json` containing examples. An example is described by:

`dict_keys(['id', 'study_id', 'subject_id', 'image_path', 'split', 'label', 'report'])`

The `report` key contains a dictionary:

`dict_keys(['findings', 'impression', 'background', 'r2gen'])`

The dataloader returns the following dictionary:
`dict_keys(['idx', 'key', 'report', 'img', 'label])` 

where key is a triple `(subject_id, study_id, image_id)`

The different tasks coded for this dataset can be found in `BaseMimic.py`
