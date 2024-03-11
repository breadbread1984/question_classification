#!/usr/bin/python3

from os.path import join
from datasets import load_dataset
from csv import reader, QUOTE_ALL
import json

def convert(orig, dst):
  outf = open(dst, 'w')
  with open(orig, 'r') as inf:
    csv = reader(inf, delimiter = '\t')
    for row in csv:
      outf.write(json.dumps({'question': row[0], 'category': int(row[1])}, ensure_ascii = False) + '\n')
  outf.close()

def load_csv(data_dir):
  convert(join(data_dir, 'train.txt'), 'train.json')
  convert(join(data_dir, 'dev.txt'), 'dev.json')
  convert(join(data_dir, 'dev.txt'), 'test.json')
  dataset = load_dataset('csv',
                         data_files = {
                           'train': 'train.json',
                           'validate': 'dev.json',
                           'test': 'test.json'},
                         column_names = ['question', 'category']
                        )
  return dataset

