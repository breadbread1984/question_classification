#!/usr/bin/python3

from absl import flags, app
from os.path import join, exists
from csv import reader
import json
import torch
from torch import device
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoPeftModelForCausalLM
from datasets import load_dataset
from create_dataset import convert

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input', default = None, help = 'path to input csv')
  flags.DEFINE_string('output', default = None, help = 'path to output csv')
  flags.DEFINE_string('ckpt', default = None, help = 'path to checkpoint')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device')

def main(unused_argv):
  assert exists(join(FLAGS.ckpt, 'adapter_config.json'))
  testset = load_dataset('json', data_files = 'test.json', field = 'data')
  model = AutoPeftModelForCausalLM.from_pretrained(FLAGS.ckpt, trust_remote_code = Tryue, device_map = "auto")
  tokenizer = AutoTokenizer.from_pretrained(model.peft_config['default'].base_model_name_or_path, trust_remote_code = True)
  model = model.to(device(FLAGS.device))
  model.eval()
  choices = ["A","B","C","D","E","F"]
  choice_tokens = [tokenizer.encode(choice, add_special_tokens = False)[0] for choice in choices]
  kwargs = {'max_length': 8192, 'num_beams': 1, 'do_sample': False, 'top_p': 1, 'temperature': 1, 'logits_processor': LogitsProcessorList()}
  of = open(FLAGS.output, 'w')
  with open(FLAGS.input, 'r') as f:
    csv = reader(f, delimiter = '\t')
    for row in csv:
      text = row[0]
      inputs = tokenizer.build_chat_input(text, history = [], role = 'user').to(device(FLAGS.device))
      eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), tokenizer.get_command("<|observation|>")]
      outputs = model(**inputs, **kwargs, eos_token_id = eos_token_id)
      logits = outputs.logits
      logits = logits[:, choice_tokens] # logits.shape = (1, 6)
      preds = logits.argmax(dim = -1).detach().cpu().numpy().item() # preds.shape = (1,)
      preds = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5}[preds]
      of.write('%s\t%d\n' % (text, preds))

if __name__ == "__main__":
  add_options()
  app.run(main)

