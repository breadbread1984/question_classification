#!/usr/bin/python3

from os.path import join
from datasets import Dataset
from csv import reader

def get_generator(csv):
  def gen():
    for row in csv:
      if len(row) == 2: yield {"question": row[0], "label": int(row[1])}
      else: yield {"question": row[0]}
  return gen

def load_dataset(data_dir = None, div = 'train'):
  assert div in {'train', 'dev', 'test'}
  with open(join(data_dir, '%s.txt' % div), 'r') as f:
    csv = reader(f, delimiter = '\t')
  return Dataset.from_generator(get_generator(csv))

