# Introduction

this project is to prediction question categories with CHATGLM3

# Usage

## Install prerequisites

```shell
python3 -m pip install -r requirements.txt
```

## Download pretrained model

```shell
python3 download_model.py
```

## Download datasets

download question dataset from [tianchi contest page](https://tianchi.aliyun.com/competition/entrance/532176) and place files under a same directory.

## Train model

```shell
python3 train.py --dataset <path/to/dataset>
```

## Inference

```shell
python3 inference.py --input <path/to/test.txt> --output pred.txt --ckpt <path/to/ckpt>
```
