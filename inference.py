#!/usr/bin/python3

from absl import flags, app
from os.path import join
from csv import reader
import json
import torch
from torch import device
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from create_dataset import convert

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input', default = None, help = 'path to input csv')
  flags.DEFINE_string('output', default = None, help = 'path to output csv')
  flags.DEFINE_string('ckpt', default = None, help = 'path to checkpoint')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device')

def main(unused_argv):
  testset = load_dataset('json', data_files = 'test.json', field = 'data')
  tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
  model = AutoModelForSequenceClassification.from_pretrained(FLAGS.ckpt)
  model = model.to(device(FLAGS.device))
  model.eval()
  of = open(FLAGS.output, 'w')
  with open(FLAGS.input, 'r') as f:
    csv = reader(f, delimiter = '\t')
    for row in csv:
      text = row[0]
      inputs = tokenizer(text, return_tensors = 'pt').to(device(FLAGS.device))
      outputs = model(**inputs)
      logits = outputs.logits
      preds = torch.argmax(logits, dim = -1)[0].detach().cpu().numpy().item()
      of.write('%s\t%d\n' % (text, preds))

if __name__ == "__main__":
  add_options()
  app.run(main)

