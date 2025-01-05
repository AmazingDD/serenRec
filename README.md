<p align="left">
<img src="seren/logo.jpg" align="center" width="60%" style="margin: 0 auto">
</p>

## Overview

serenRec is a Python toolkit developed for sequential recommendation baselines and experiments.

## How to Run

```
python main.py -use_cuda -gpu_id=0 -model=sasrec
```

## Implemented Algorithms

| **Model** | **Publication** |
|-----------|-----------------|
| Pop | A re-visit of the popularity baseline in recommender systems (SIGIR'2020) |
| Caser | Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding (WSDM'2018) |
| SASRec | Self-Attentive Sequential Recommendation (ICDM'2018) |
| SRGNN | Session-based Recommendation with Graph Neural Networks (AAAI'2019) |
| STAMP | STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation (KDD'2018) |
| GRU4Rec | Improved Recurrent Neural Networks for Session-based Recommendations (RecSys'2016) |
| FMLP | Filter-enhanced MLP is All You Need for Sequential Recommendation (WWW'2022) |
| LRURec | Linear Recurrent Units for Sequential Recommendation (WSDM'2024) |
| BSARec | An Attentive Inductive bias for Sequential Recommendation Beyond the Self-Attention (AAAI'2024) |

## TODO List

- [x] More baselines
- [ ] interface for more datasets
<!-- - [ ] find some long tail baselines -->
