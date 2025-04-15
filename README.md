# LIFT+: Lightweight Fine-Tuning for Long-Tail Learning

This is the source code for the paper "LIFT+: Lightweight Fine-Tuning for Long-Tail Learning".

## Installation

This code repository is implemented in Python. For environment setup, we recommend installing Python via [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install). To install the required Python dependencies, please run:

```sh
conda install pytorch torchvision pytorch-cuda -c pytorch -c nvidia
conda install scikit-learn yacs tensorboard -c conda-forge
pip install openai-clip timm
```

## Hardware

Most experiments can be reproduced using a single GPU with 24GB of memory.

- To further reduce the GPU memory cost, gradient accumulation is recommended. Please refer to the [Detailed Usage](#detailed-usage) for instructions.

## Quick Start on the CIFAR-100-LT dataset

```bash
# run LIFT+ on CIFAR-100-LT (using CLIP)
python main.py -d cifar100_ir100 -b clip_vit_b16 -m lift+
python main.py -d cifar100_ir50 -b clip_vit_b16 -m lift+
python main.py -d cifar100_ir10 -b clip_vit_b16 -m lift+
```

By running the above command, you can automatically download the CIFAR-100 dataset and run the method (LIFT).

## Running on Large-scale Long-tailed Datasets

### Prepare the Dataset

Download the dataset [ImageNet](http://image-net.org/index), [Places](http://places2.csail.mit.edu/download.html), and [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018).

Put files in the following locations and change the path in the data configuration files in [configs/data](configs/data):

- ImageNet

```
Path/To/Dataset
├─ train
│  ├─ n01440764
|  |  ├─ n01440764_18.JPEG
|  |  └─ ......
│  └─ ......
└─ val
   ├─ n01440764
   |  ├─ ILSVRC2012_val_00000293.JPEG
   |  └─ ......
   └─ ......
```

- Places

```
Path/To/Dataset
├─ train
│  ├─ airfield
|  |  ├─ 00000001.jpg
|  |  └─ ......
│  └─ ......
└─ val
   ├─ airfield
   |  ├─ Places365_val_00000435.jpg
   |  └─ ......
   └─ ......
```

- iNaturalist 2018

```
Path/To/Dataset
└─ train_val2018
   ├─ Actinopterygii
   |  ├─ 2229
   |  |  ├─ 2c5596da5091695e44b5604c2a53c477.jpg
   |  |  └─ ......
   |  └─ ......
   └─ ......
```

### Reproduction

To reproduce the main result in the paper, please run

```bash
# LIFT+ on ImageNet-LT (using CLIP)
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+

# LIFT+ on Places-LT (using CLIP)
python main.py -d places_lt -b clip_vit_b16 -m lift+

# LIFT+ on Places-LT (using ImageNet-21K pre-trained ViT)
python main.py -d places_lt -b in21k_vit_b16 -m lift+

# LIFT+ on iNaturalist 2018 (using CLIP)
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15

# LIFT+ on iNaturalist 2018 (using ImageNet-21K pre-trained ViT)
python main.py -d inat2018 -b in21k_vit_b16 -m lift+ num_epochs 15

# LIFT on CIFAR-100-LT (using CLIP)
python main.py -d cifar100_ir100 -b clip_vit_b16 -m lift+
python main.py -d cifar100_ir50 -b clip_vit_b16 -m lift+
python main.py -d cifar100_ir10 -b clip_vit_b16 -m lift+

# LIFT with pre-trained ViT from ImageNet-21K
python main.py -d cifar100_ir100 -b in21k_vit_b16 -m lift+
python main.py -d cifar100_ir50 -b in21k_vit_b16 -m lift+
python main.py -d cifar100_ir10 -b in21k_vit_b16 -m lift+
```

For other experiments, please refer to [scripts](scripts) for reproduction commands.

### Detailed Usage

To train and test the proposed method on more settings, run

```bash
python main.py -d [data] -b [backbone] -m [method] [options]
```

The `[data]` can be the name of a .yaml file in [configs/data](configs/data), including `imagenet_lt`, `places_lt`, `inat2018`, `cifar100_ir100`, `cifar100_ir50`, `cifar100_ir10`, etc.

The `[backbone]` can be the name of a .yaml file in [configs/backbone](configs/backbone), including `clip_vit_b16`, `in21k_vit_b16`, `clip_vit_l14`, `clip_rn50`, etc.

The `[method]` can be the name of a .yaml file in [configs/method](configs/method), including `lift+`, `lift`, `zs`, `lp`, `fft`, `aft`, etc.

Moreover, `[options]` can facilitate modifying the configuration options in [utils/config.py](utils/config.py). The following are some examples.

- To specify the root path of datasets, add `root Path/To/Datasets`.
- To change the output directory, add an option like `output_dir NewExpDir`. Then the results will be saved in `output/NewExpDir`.
- To assign a single GPU (for example, GPU 0), add an option like `gpu 0`.
- To apply gradient accumulation, add `accum_step XX`. This can further reduce GPU memory costs. Note that `XX` should be a divisor of `batch_size`.
- To test an existing model, add `test_only True`. This option will test the model trained by your configuration file. To test another model, add an option like `model_dir output/AnotherExpDir`.

All of the reproduction commands are provided in [scripts](scripts). You can refer to the presented commands for examples.

## Acknowledgment

We thank the authors for the following repositories for code reference:
[[OLTR]](https://github.com/zhmiao/OpenLongTailRecognition-OLTR), [[Classifier-Balancing]](https://github.com/facebookresearch/classifier-balancing), [[Dassl]](https://github.com/KaiyangZhou/Dassl.pytorch), [[CoOp]](https://github.com/KaiyangZhou/CoOp), [[LIFT]](https://github.com/shijxcs/LIFT).
