# Conditional Generative Adversarial Network (CGAN)

## Introduction
Conditional Generative Adversarial Network (CGAN)는 특정 조건을 만족하는 새로운 데이터를 생성을 목표로하는 생성 모델입니다.
CGAN은 vanilla GAN에 condition을 추가하여 학습한 모델입니다. MNIST의 경우 0 ~ 9까지의 label을 조건으로 주어 CGAN이 학습하면서 각각의 label에 해당하는 데이터를 생성하는 순차적인 결과의 변화를 gif 형식의 파일로 가시화할 수 있습니다. 또한 학습된 모델이 생성한 데이터의 질을 확인하기 위하여 [Fréchet Inception Distance (FID) score](https://github.com/mseitzer/pytorch-fid)를 계산할 수 있습니다(출처: https://github.com/mseitzer/pytorch-fid). CGAN에 대한 설명은 [Conditional Generative Adversarial Network (CGAN)](https://ljm565.github.io/contents/CGAN1.html)를 참고하시기 바랍니다.
<br><br><br>

## Supported Models
### Conditional GAN
`nn.Linear`를 사용한 vanilla CGAN 구현되어 있습니다.
<br><br><br>

## Base Dataset
* 튜토리얼로 사용하는 기본 데이터는 [Yann LeCun, Corinna Cortes의 MNIST](http://yann.lecun.com/exdb/mnist/) 데이터입니다.
* `config/config.yaml`에 학습 데이터의 경로를 설정하여 사용자가 가지고 있는 custom 데이터도 학습 가능합니다.
다만 `src/utils/data_utils.py`에 custom dataloader 코드를 구현해야할 수도 있습니다.
<br><br><br>

## Supported Devices
* CPU, GPU, multi-GPU (DDP), MPS (for Mac and torch>=1.12.0)
<br><br><br>

## Quick Start
```bash
python3 src/run/train.py --config config/config.yaml --mode train
```
<br><br>

## Project Tree
본 레포지토리는 아래와 같은 구조로 구성됩니다.
```
├── configs                         <- Config 파일들을 저장하는 폴더
│   └── *.yaml
│
└── src      
    ├── models
    |   └── cgan.py                  <- CGAN 모델 파일
    |
    ├── run                   
    |   ├── cal_fid.py              <- FID score 계산 실행 파일
    |   ├── train.py                <- 학습 실행 파일
    |   └── validation.py           <- 학습된 모델 평가 실행 파일
    | 
    ├── tools    
    |   ├── pytorch_fid             <- FID score를 계산하기 위한 코드
    |   |   ├── fid_score.py
    |   |   └── inception.py
    |   |
    |   ├── model_manager.py          
    |   └── training_logger.py      <- Training logger class 파일
    |
    ├── trainer                 
    |   ├── build.py                <- Dataset, dataloader 등을 정의하는 파일
    |   └── trainer.py              <- 학습, 평가, FID score 계산 class 파일
    |
    └── uitls                   
        ├── __init__.py             <- Logger, 버전 등을 초기화 하는 파일
        ├── data_utils.py           <- Custom dataloader 파일
        ├── filesys_utils.py       
        └── training_utils.py     
```
<br><br>

## Tutorials & Documentations
CGAN 모델 학습을 위해서 다음 과정을 따라주시기 바랍니다.
1. [Getting Started](./1_getting_started_ko.md)
2. [Data Preparation](./2_data_preparation_ko.md)
3. [Training](./3_trainig_ko.md)
4. ETC
   * [Evaluation](./4_model_evaluation_ko.md)
   * [FID Calculation](./5_calculate_fid_ko.md)
<br><br><br>


## Training Results
* CGAN 학습 결과<br><br>
<img src="figs/generation_gif.gif" width="50%"><br><br>
<img src="figs/RealAndFake.png" width="100%"><br><br>
<br><br><br>