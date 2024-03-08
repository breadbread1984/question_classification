#!/usr/bin/python3

from absl import flags, app
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import evaluate

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint directory')

def main(unused_argv):
  tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-3')
  model = AutoModelForCausalLM.from_pretrained('THUDM/chatglm-3')
  choices = ['A', 'B', 'C', 'D', 'E', 'F']
  choice_tokens = [tokenizer.encode(choice, add_special_tokens = False)[0] for choice in choices]
  training_args = TrainingArguments(output_dir = FLAGS.ckpt)
  def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits[:, choice_tokens], axis = -1)
    return metric.compute(predictions = predictions, references = labels)
  
