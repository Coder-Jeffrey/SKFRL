# Source code of SKFRL
## The main requirements. You may need to download some other requirements depending on your environment
```
torch == 1.13.1
python == 3.7.16
transformers == 4.24.0
scikit-learn == 1.0.2
torchmetrics == 0.3.2
pytorch-lightning == 1.3.7
spacy == 3.0.6
```
## Datasets
### EURLEX-57K
http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip
### CMU Book Summary
http://www.cs.cmu.edu/~dbamman/booksummaries.html

## ConceptNet
https://github.com/commonsense/conceptnet5/wiki/Downloads

## The BERT pre-trained model.
https://huggingface.co/

## External Knowledge
build_kg.py is used to find the concepts triple for datasets.
TransE(a public method) is used to train the external knowledge.
Next, use PPMI.py to enhance the external knowledge representation.

## Long text segmentation

text_segmentation.py designs a dynamic programming method for long text segmentation.

## Run
* cd src

* run train.py
