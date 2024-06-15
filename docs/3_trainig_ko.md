# Training CGAN
여기서는 CGAN 모델을 학습하는 가이드를 제공합니다.

### 1. Configuration Preparation
CGAN 모델을 학습하기 위해서는 Configuration을 작성하여야 합니다.
<b>그리고 사용할 데이터는 이미지의 class labeling이 되어있어야 합니다.</b>
Configuration에 대한 option들의 자세한 설명 및 예시는 다음과 같습니다.

```yaml
# base
seed: 0
deterministic: True

# environment config
device: cpu     # examples: [0], [0,1], [1,2,3], cpu, mac: mps... 

# project config
project: outputs/CGAN
name: MNIST

# image setting config
height: 28
width: 28
color_channel: 1
convert2grayscale: False

# data config
workers: 0               # Don't worry to set worker. The number of workers will be set automatically according to the batch size.
MNIST_train: True        # if True, MNIST will be loaded automatically.
class_num: 10            # Number of image label classes.
label_dim: 32            # Class information embedding dimension.
MNIST:
    path: data/
    MNIST_valset_proportion: 0.2      # MNIST has only train and test data. Thus, part of the training data is used as a validation set.
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null

# train config
batch_size: 128
epochs: 1
lr: 0.0002
hidden_dim: 256
noise_init_size: 128

# logging config
common: ['train_loss_d', 'train_loss_g', 'validation_loss_d', 'validation_loss_g', 'd_x', 'd_g1', 'd_g2']
```

### 2. Training
#### 2.1 Arguments
`src/run/train.py`를 실행시키기 위한 몇 가지 argument가 있습니다.
* [`-c`, `--config`]: 학습 수행을 위한 config file 경로.
* [`-m`, `--mode`]: [`train`, `resume`] 중 하나를 선택.
* [`-r`, `--resume_model_dir`]: mode가 `resume`일 때 모델 경로. `{$project}/{$name}`까지의 경로만 입력하면, 자동으로 `{$project}/{$name}/weights/`의 모델을 선택하여 resume을 수행.
* [`-l`, `--load_model_type`]: [`loss`, `last`] 중 하나를 선택.
    * `loss`(default): Valdiation loss가 최소일 때 모델을 resume.
    * `last`: Last epoch에 저장된 모델을 resume.
* [`-p`, `--port`]: (default: `10001`) DDP 학습 시 NCCL port.


#### 2.2 Command
`src/run/train.py` 파일로 다음과 같은 명령어를 통해 CGAN 모델을 학습합니다.
```bash
# training from scratch
python3 src/run/train.py --config configs/config.yaml --mode train

# training from resumed model
python3 src/run/train.py --config config/config.yaml --mode resume --resume_model_dir {$project}/{$name}
```
모델 학습이 끝나면 `{$project}/{$name}/weights`에 체크포인트가 저장되며, `{$project}/{$name}/args.yaml`에 학습 config가 저장됩니다.