#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import join, exists
from shutil import rmtree
from datasets import Dataset
from csv import reader
import json

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset directory')
  flags.DEFINE_string('output', default = 'dataset', help = 'path to output')

def generate_dataset(data_dir, div = 'train', output = "output.csv"):
  assert div in {'train', 'dev', 'test'}
  prompt = '以下是糖尿病问题分类单项选择题，请选出正确类别。\n\n%s\n\nA. 诊断\nB. 治疗\nC. 常识\nD. 健康生活方式\nE. 流行病学\nF. 其他\n答案：'
  with open(join(data_dir, '%s.txt' % div), 'r') as f:
    csv = reader(f, delimiter = '\t')
    samples = list()
    for row in csv:
      sample = {"conversations": [
        {
          "role": "user",
          "content": prompt % row[0]
        },
        {
          "role": "assistant",
          "content": "ABCDEF"[int(row[1])]
        }
      ]}
      samples.append(sample)
  with open(output, 'w') as f:
    f.write(json.dumps(samples))

def main(unused_argv):
  if exists(FLAGS.output): rmtree(FLAGS.output)
  mkdir(FLAGS.output)
  generate_dataset(FLAGS.dataset, 'train', output = join(FLAGS.output, 'train.json'))
  generate_dataset(FLAGS.dataset, 'dev', output = join(FLAGS.output, 'val.json'))

if __name__ == "__main__":
  add_options()
  app.run(main)

