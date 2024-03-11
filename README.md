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

## Download & create datasets

download question dataset from [tianchi contest page](https://tianchi.aliyun.com/competition/entrance/532176) and place files under a same directory.

```shell
python3 create_dataset.py --dataset <path/to/raw/dataset> --output <path/to/processed/dataset>
```

## Download source & train ChatGLM3

```shell
git clone https://github.com/THUDM/ChatGLM3
cd ChatGLM3/finetune_demo
```

change **finetune_demo/finetune_hf.py**:151 to

```python
default_factory=Seq2SeqTrainingArguments(output_dir='./output')
```

train with the following command

```shell
python finetune_hf.py <path/to/processed/dataset> THUDM/chatglm3-6b configs/lora.yaml
```

## Inference

```shell
python3 inference.py --input <path/to/test.txt> --output pred.txt --ckpt <path/to/ckpt>
```
