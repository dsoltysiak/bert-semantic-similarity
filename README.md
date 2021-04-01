# BERT Semantic Text Similarity

This repository contains fine-tuned [BERT](https://github.com/google-research/bert) model for Semantic Text Similarity (STS). It's based on the BERT-base-uncased model and the implementation is done using Tensorflow and Keras. This model aims to find the type of semantic similarity given two similar sentences.

The dataset used for this project is [The Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/). It consists of english sentence-pairs with the labels entailment, contradiction, and neutral.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install needed libraries. 

```bash
pip install numpy
pip install pandas
pip install transformers
pip install tensorflow-gpu
pip install tensorflow-addons
```
## Dependencies

```bash
numpy==1.19.5
pandas==1.1.5
python==3.7.1
transformers==4.4.2
tensorflow-gpu==2.4.1
tensorflow-addons==0.12.1
```

## Evaluation of results

The predictive performance of the model can be characterized by the table below.

| Metrics       | Dev Set        | Test Set  |
| ------------- |-------------| -----|
| Accuracy      | 88.09 | 87.36 |
| Precision     | 88.97      |   88.21 |
| Recall | 87.31      |    86.34 |
| F1 | 88.05 | 87.35 |
| AUC | 96.99 | 96.54 |



## References
* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

## License
[MIT](https://choosealicense.com/licenses/mit/)