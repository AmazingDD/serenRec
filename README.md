# SSR

## Overview

- This repository also contains the related code for our article published in IJCAI'2025, "Cost-Effective On-Device Sequential Recommendation with Spiking Neural Networks", located in `seren.recommender.ssr`. The name **SSR** stands for **S**pike-wise **S**equential **R**ecommendation.

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

## How to Run

Ensure you have a CUDA environment to accelerate, since the deep-learning models could be based on it.

a quick start tutorial with ML-1M toy implementation

To quickly get the testing results, please implement:
```
python main.py -use_cuda -gpu_id=0 -dataset=ml-1m -model=ssr
```

To use the automatic TPE tuning method to get a better testing result, please implement
```
python main.py -use_cuda -gpu_id=0 -dataset=ml-1m -model=ssr -tune -nt=20
```
`-tune -nt` will allow the code to search the best hyperparameter settings 20 times with the maximum target `MRR@10`

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
