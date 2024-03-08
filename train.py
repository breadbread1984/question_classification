#!/usr/bin/python3

from absl import flags, app
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import evaluate
from create_dataset import load_dataset

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint directory')

def main(unused_argv):
  tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-3')
  model = AutoModelForCausalLM.from_pretrained('THUDM/chatglm-3')
  choices = ['A', 'B', 'C', 'D', 'E', 'F']
  choice_tokens = [tokenizer.encode(choice, add_special_tokens = False)[0] for choice in choices]
  training_args = TrainingArguments(output_dir = FLAGS.ckpt, evaluation_strategy = "epoch")
  metric = evaluate.load('accuracy')
  def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits[:, choice_tokens], axis = -1)
    return metric.compute(predictions = predictions, references = labels)
  trainer = Trainer(model = model, args = training_args, train_dataset = trainset, eval_dataset = evalset, compute_metrics = compute_metrics)
  trainer.train()

if __name__ == "__main__":
  add_options()
  app.run(main)

