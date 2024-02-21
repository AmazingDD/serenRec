<p align="left">
<img src="seren/logo.jpg" align="center" width="60%" style="margin: 0 auto">
</p>

## Overview

serenRec is a python toolkit developed for sequential-/session-based recommendation baselines and experiments.

## How to Run

```
python main.py -use_cuda -gpu_id=0 -model=sasrec
```

## Implemented Algorithms

| **Model** | **Publication** |
|-----------|-----------------|
| Session-Pop | A re-visit of the popularity baseline in recommender systems |
| Session-MF | Matrix factorization techniques for recommender systems |
| Caser | Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding |
| SASRec | Self-Attentive Sequential Recommendation. |
| SRGNN | Session-based Recommendation with Graph Neural Networks |
| STAMP | STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation |
| GRU4Rec | Improved Recurrent Neural Networks for Session-based Recommendations |
| FMLP | Filter-enhanced MLP is All You Need for Sequential Recommendation |

## TODO List

- [ ] More baselines
