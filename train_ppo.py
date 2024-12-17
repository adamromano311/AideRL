# Imports
from google.colab import drive
import random
import json
import os
import torch
import glob
import wandb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl import RewardTrainer, RewardConfig
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from datasets import load_dataset, Dataset
from peft import LoraConfig, TaskType
from huggingface_hub import login

def read_json_file(file_path):
  """Reads a JSON file and returns the data as a Python object.

  Args:
    file_path: The path to the JSON file.

  Returns:
    A Python object (usually a dictionary or a list) representing the JSON data.
  """
  with open(file_path, 'r') as file:
    data = json.load(file)
  return data

def read_txt_file(file_path):
    """Reads a txt file and returns its content.

    Args:
      file_path: Path to the txt file.

    Returns:
      The entire content of the file as a string.
    """
    with open(file_path, 'r') as file:
        file_content = file.read()
    return file_content

def get_all_files(folder_path):
  """Gets a list of all files in a folder.

  Args:
    folder_path: The path to the folder.

  Returns:
    A list of strings, where each string is the name of a file in the folder.
  """
  files = []
  for item in os.listdir(folder_path):
    item_path = os.path.join(folder_path, item)
    if os.path.isfile(item_path):
      files.append(item)
  return files

def extract_text(filename):
    """
    Extracts the text from the first '# Introduction' until the first 'INFO: response:' line.

    Args:
        filename: The name of the file to extract text from.

    Returns:
        A string containing the extracted text or None if the specified markers
        are not found.
    """

    with open(filename, 'r') as f:
        lines = f.readlines()

    start_marker = '# Introduction'
    end_marker = 'INFO: response:'
    extracted_text = []
    start_found = False

    for line in lines:
        if start_marker in line:
            start_found = True
            extracted_text.append(start_marker)
        elif end_marker in line and start_found:
            break
        elif start_found:
            extracted_text.append(line)

    if extracted_text:
        return ''.join(extracted_text)
    else:
        return None

login()
wandb.login(key="0444d0091009eba5725ae52fd071f747971c78d7")

base_dir = '/content/AideRL/'

# Load all the journal files for the teacher GPT-4o model
run_dir = os.path.join(base_dir, 'get_prompt_run')
competitions = [os.path.basename(x) for x in glob.glob(os.path.join(run_dir,'*'))]
prompts = []
for comp in competitions:
  verbose_dir = os.path.join(run_dir, comp, 'logs', 'aide.verbose.log')
  extracted_content = extract_text(verbose_dir)
  if extracted_content:
    prompts.append({"prompt":extracted_content})
  else:
    print(f"The specified markers were not found in {verbose_dir}")

prompts_dataset = Dataset.from_list(prompts)

# Load the models needed for reinforcement learning
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
if tokenizer.chat_template is None:
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
model = AutoModelForCausalLM.from_pretrained(model_name)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)
trained_reward_model = AutoModelForSequenceClassification.from_pretrained("/content/trained_reward_model_llama1B")

################
# Dataset
################
eval_samples = 10
train_dataset = prompts_dataset.select(range(len(prompts_dataset) - eval_samples))
eval_dataset = prompts_dataset.select(range(len(prompts_dataset) - eval_samples, len(prompts_dataset)))
dataset_text_field = "prompt"

def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field],
            padding=False,
            truncation=True,  # Add truncation
            max_length=2048,   # Limit sequence length (adjust as needed)
        )
        return {"input_ids": outputs["input_ids"]}

    return dataset.map(
        tokenize,
        batched=True,
        batch_size=1,
        remove_columns=dataset.column_names,
    )

# Compute that only on the main process for faster data processing.
# see: https://github.com/huggingface/trl/pull/1255
train_prompt_dataset = prepare_dataset(train_dataset, tokenizer)
eval_prompt_dataset = prepare_dataset(eval_dataset, tokenizer)

################
# Training
################

ppo_config = {"mini_batch_size": 1, "batch_size": 1, "output_dir": 'content/'}
config = PPOConfig(**ppo_config)
trainer = PPOTrainer(
    config,
    tokenizer,
    policy=model,
    ref_policy=ref_model,
    reward_model=trained_reward_model,
    value_model=trained_reward_model,
    train_dataset=train_prompt_dataset,
    eval_dataset=eval_prompt_dataset,
)

# Train Llama model with RL
trainer.train()

# Save the trained Llama model
trainer.save_model("/content/trained_model_llama1B")



