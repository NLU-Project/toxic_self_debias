# toxic_self_debias

This repo contains the code for our paper: [Applying Self Debiasing 
Techniques to Toxic Language Detection Language Models](https://github.com/NLU-Project/toxic_self_debias/blob/main/Applying%20Self%20Debiasing%20Techniques%20to%20Toxic%20Language%20Detection%20Models.pdf). This is a project for
[NYU DS-GA 1012](https://docs.google.com/document/d/e/2PACX-1vRydPvLp9tNw1-45pp6IIl-jppX-tUfu0TQDVXRAiGA3CjIuJzBTzJo7cerQV08K8FqfUOYHBCPAggx/pub)
Natural Language Understanding. The course requires students to contribute to
the field of NLU with new research. In particular, our work focuses on combining
the work of [Lees et al., 2021](https://github.com/XuhuiZhou/Toxic_Debias), and
the self-debiasing method outlined by [Utama et al., 2020](https://arxiv.org/abs/2009.12303)
. Currently we have created biased versions of the models, but soon we will be
encorporating code from the [self-debiasing repo](https://github.com/UKPLab/emnlp2020-debiasing-unknownit)
to attempt to improve Toxic Language Detection for minority classified text.
The code has been edited to enable fine-tuning of BERT and RoBERTa on the different
data sets we are using, and to evaluate on a separate dataset.

<!---
Sara Price, Pavel Gladkevich, David May, Pedro Galarza 2022
-->
## Overview
### Data
This repo contains code to detect toxic language with BERT/RoBERTa
Our experiments mainly focus on finetuning either on the dataset from 
["Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior"](https://ojs.aaai.org/index.php/ICWSM/article/view/14991) aka Founta, or on
["Nuanced Metrics for Measuring Unintended Bias
with Real Data for Text Classification"](https://arxiv.org/pdf/1903.04561.pdf) aka Civil Comments.



### Code
Our implementation exists in the `.\src` folder. The `run_toxic.py` file
organize the classifier, and the `modeling_roberta_debias.py` builds the
ensemble-based model.

## Setup 

### Dependencies

We require pytorch(specifiy version) and transformers(specify version)
Additional requirements are are in
`requirements.txt` (Currently not accurate, has excessive non-needed)

### Work in progress the below needs to be edited

### Data

* You can find the index of the training data with different data selection
  methods in `data/founta/train`
* You can find a complete list of entries of data that we need for experiments
  in `data/demo.csv`
* Out-of-distribution (OOD) data, the two OOD datasets we use are publicly
  available:
    * ONI-adv: This dataset is the test set of the work ["Build it Break it Fix
it for Dialogue Safety: Robustness from Adversarial Human
Attack"](https://www.aclweb.org/anthology/D19-1461/)
    * User-reported: This dataset is from the work ['User-Level Race and Ethnicity Predictors from Twitter Text'](https://www.aclweb.org/anthology/C18-1130/)

* Our word list for lexical bias is in the file: `./data/word_based_bias_list.csv`
* Since we do not encourage building systems based on our relabeling dataset,
  we decide not to release the relabeling dataset publicly. For research purpose, please
  contact the first author for the access of the dataset.

## Experiments

### Measure Dataset Bias
Run 
```python 
python ./tools/get_stats.py /location/of/your/data_file.csv

```
To obtain the Peasonr correlation between toxicity and Tox-Trig words/ aav
probabilities.

### Fine-tune a Vanilla RoBERTa
Run 
```bash
sh run_toxic.sh 
```

### Fine-tune a Ensemble-based RoBERTa
Run 
```bash
sh run_toxic_debias.sh
```

You need to obtain the bias-only model first in order to train the ensemble
model. Feel free to use files we provided in the folder `tools`.

### Model Evaluation & Measuring Models' Bias

You can use the same fine-tuning script to obtain predictions from models. 

The measuring bias script takes the predictions as input and output models'
performance and lexical/dialectal bias scores. The script is available in the
`src` folder.
