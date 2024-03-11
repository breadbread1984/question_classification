#!/usr/bin/python3

from absl import flags, app
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from create_dataset import load_csv
import numpy as np

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset directory')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to save checkpoint directory')
  flags.DEFINE_integer('seed', default = 42, help = 'random seed')

def main(unused_argv):
  tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
  dataset = load_csv(FLAGS.dataset)
  tokenized_datasets = dataset.map(lambda x: tokenizer(x["text"], padding = 'max_length', truncation = True), batched = True)
  trainset = tokenized_datasets["train"].shuffle(seed = FLAGS.seed)
  valset = tokenized_datasets['test'].shuffle(seed = FLAGS.seed)

  model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-chinese', num_labels = 6)
  training_args = TrainingArguments(output_dir = FLAGS.ckpt)
  metric = evaluate.load('accuracy')
  def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    return metric.compute(predictions = predictions, references = labels)
  trainer = Trainer(model = model, args = training_args, train_dataset = trainset, eval_dataset = valset, compute_metrics = compute_metrics)
  trainer.train()

if __name__ == "__main__":
  add_options()
  app.run(main)

