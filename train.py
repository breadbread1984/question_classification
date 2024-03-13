#!/usr/bin/python3

from absl import flags, app
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from create_dataset import load_csv
import numpy as np

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset directory')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to save checkpoint directory')
  flags.DEFINE_float('lr', default = 1e-5, help = 'learning rate')
  flags.DEFINE_integer('batch_size', default = 32, help = 'batch size')
  flags.DEFINE_integer('epoch', default = 10, help = 'epoch')
  flags.DEFINE_integer('seed', default = 42, help = 'random seed')

def main(unused_argv):
  login(token = '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')
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
  model = AutoModelForSequenceClassification.from_pretrained('meta-llama/Llama-2-7b-hf', num_labels = 6)
  optimizer = AdamW(model.parameters(), lr = FLAGS.lr)
  dataset = load_csv(FLAGS.dataset)
  ce = CrossEntropyLoss()
  metric = evaluate.load('accuracy')
  for epoch in range(FLAGS.epoch):
    # 1) train phase
    model.train()
    for sample in dataset['train']:
      optimizer.zero_grad()
      encoded_input = tokenizer(sample['text'], padding = True, truncation = True, return_tensors = 'pt')
      outputs = model(**encoded_input)
      logits = outputs.logits
      loss = ce(logits, sample['label'])
      loss.backward()
      optimizer.step()
    model.save_pretrained(FLAGS.ckpt)
    tokenizer.save_pretrained(FLAGS.ckpt)
    # 2) eval phase
    model.eval()
    preds = list()
    labels = list()
    for sample in dataset['validate']:
      encoded_input = tokenizer(sample['text'], padding = True, truncation = True, return_tensors = 'pt')
      outputs = model(**encoded_input)
      logits = outputs.logits
      pred = np.argmax(logits, axis = -1)
      preds.append(pred)
      labels.append(sample['label'])
    print('evaluation: accuracy %f' % metric.compute(predictions = preds, references = labels))

if __name__ == "__main__":
  add_options()
  app.run(main)

