<p align="left">
<img src="assets/logo.jpg" align="center" width="60%" style="margin: 0 auto">
</p>

## Overview

- **serenRec** is a Python toolkit developed to benchmark sequential recommendation baselines and experiments. The name **SEREN** stands for **SE**quential **RE**commendatio**N**.

<!-- <p align="center">
    <img width=70% src="./assets/ssr-overview.png" alt="Image 1" style="margin: 0 40px;">
</p> -->

## Requirements

```
optuna==3.5.0
torch==2.0.1
spikingjelly==0.0.0.0.14
numpy==1.23.5
pandas==1.5.3
```

## Datasets

make sure all data files required are placed in the correct corresponding path:

```
│movielens/
├──ml-1m/
│  ├── ratings.dat
│amazon/
│  ├── Digital_Music.csv
│  ├── Video_Games.csv
|  ├── Arts_Crafts_and_Sewing.csv
│steam/
│  ├── steam_reviews.json.gz
│retail/
│  ├── events.csv
```

All datasets can be downloaded by following links:

<!-- - [ML-25M](https://files.grouplens.org/datasets/movielens/ml-25m.zip) `ml-25m` -->
- [Movielens-1M](https://github.com/AmazingDD/SSR/blob/main/movielens/ml-1m/ratings.dat) `ml-1m`
- [Amazon-Digital Music](https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Digital_Music.csv) `music`
- [Amazon-Video Games](https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games.csv) `video`
- [Amazon-Arts, Crafts, Sewing](https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Arts_Crafts_and_Sewing.csv) `arts`
- [Steam](http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz) `steam`
- [retailrocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) `retail`
<!-- - [yoochoose](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015) `yoochoose` -->

## How to Run

Ensure you have a CUDA environment to accelerate, since the deep-learning models could be based on it.

a quick start tutorial with ML-1M toy implementation

To quickly get the testing results, please implement:
```
python main.py -use_cuda -gpu_id=0 -dataset=ml-1m -model=gru4rec
```

To use the automatic TPE tuning method to get a better testing result, please implement
```
python main.py -use_cuda -gpu_id=0 -dataset=ml-1m -model=gru4rec -tune -nt=20
```
`-tune -nt` will allow the code to search the best hyperparameter settings 20 times with the maximum target `MRR@10`


## Implemented Sequential Recommendation Baselines

| **Model** | **Publication** |
|-----------|-----------------|
| POP | A revisit of the popularity baseline in recommender systems (SIGIR'2020) |
| GRU4REC | Improved Recurrent Neural Networks for Session-based Recommendations (RecSys'2016) |
| NARM | Neural Attentive Session-based Recommendation (CIKM 2017) | 
| CASER | Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding (WSDM'2018) |
| SASREC | Self-Attentive Sequential Recommendation (ICDM'2018) |
| STAMP | STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation (KDD'2018) |
| SRGNN | Session-based Recommendation with Graph Neural Networks (AAAI'2019) |
| FMLP | Filter-enhanced MLP is All You Need for Sequential Recommendation (WWW'2022) |
| LRUREC | Linear Recurrent Units for Sequential Recommendation (WSDM'2024) |
| BSAREC | An Attentive Inductive bias for Sequential Recommendation Beyond the Self-Attention (AAAI'2024) |

## Cite

Please cite the following paper if you find our work contributes to yours in any way:

```
@inproceedings{TBD,
  title={Cost-Effective On-Device Sequential Recommendation with Spiking Neural Networks},
  author={Di, Yu and Changze, Lv and Linshan, Jiang and Xin, Du and Qing, Yin and Wentao, Tong and Shuiguang, Deng and Xiaoqing, Zheng},
  booktitle={Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}},
  year={2025}
}
``` 
