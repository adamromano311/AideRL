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

# Combine prompt, random non-buggy teacher model solution, 2 (or more ? )random buggy student model solutions for each competition
def create_preference_dataset(model1_nodes, model2_nodes):
    """
    Creates a preference dataset for RLHF from two journal files.

    Args:
        model1_nodes: These nodes will come from gpt model. We will combine gpt nodes within itself as well.
        model2_nodes: These nodes will come from llama.

    Returns:
        A list of dictionaries, where each dictionary represents a preference pair:
        {"chosen": <chosen_code>, "rejected": <rejected_code>}
    """

    preference_data = []

    # Pair nodes within the same model based on competition and step
    for i in range(len(model1_nodes) - 1):
      for j in range(i + 1, len(model1_nodes)):
        node1 = model1_nodes[i]
        node2 = model1_nodes[j]
        # Preference logic:
        if node1['is_buggy'] == False and node2['is_buggy'] == True:
          preference_data.append({"chosen": node1['model_output'], "rejected": node2['model_output']})
        elif node1['is_buggy'] == True and node2['is_buggy'] == False:
          preference_data.append({"chosen": node2['model_output'], "rejected": node1['model_output']})
        elif node1['is_buggy'] == False and node2['is_buggy'] == False:
          if node1['metric']['value'] < node2['metric']['value']:
            preference_data.append({"chosen": node1['model_output'], "rejected": node2['model_output']})
          else:
            preference_data.append({"chosen": node2['model_output'], "rejected": node1['model_output']})

    # Pair nodes across different models based on competition and step
    for node1 in model1_nodes:
      count = 0
      while count < 5: # combining with 5 random nodes from Llama nodes
        random_idx = random.randint(0, len(model2_nodes) - 1)
        node2 = model2_nodes[random_idx]
        # Preference logic:
        if node1['is_buggy'] == False and node2['is_buggy'] == True:
          preference_data.append({"chosen": node1['model_output'], "rejected": node2['model_output']})
          count+=1
        elif node1['is_buggy'] == True and node2['is_buggy'] == False:
          preference_data.append({"chosen": node2['model_output'], "rejected": node1['model_output']})
          count+=1
        elif node1['is_buggy'] == False and node2['is_buggy'] == False:
          if node1['metric']['value'] < node2['metric']['value']:
            preference_data.append({"chosen": node1['model_output'], "rejected": node2['model_output']})
            count+=1
          else:
            preference_data.append({"chosen": node2['model_output'], "rejected": node1['model_output']})
            count+=1
        else:
          break

    return preference_data

def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        # Tokenize chosen and rejected sequences separately
        chosen_outputs = tokenizer(
            element["prompt"],
            element["chosen"],
            padding=False,
            truncation=True,  # Add truncation
            max_length=5020
        )
        rejected_outputs = tokenizer(
            element["prompt"],
            element["rejected"],
            padding=False,
            truncation=True,  # Add truncation
            max_length=5020
        )

        # Return the expected keys
        return {
            "input_ids_chosen": chosen_outputs["input_ids"],
            "attention_mask_chosen": chosen_outputs["attention_mask"],
            "input_ids_rejected": rejected_outputs["input_ids"],
            "attention_mask_rejected": rejected_outputs["attention_mask"],
        }

    return dataset.map(
        tokenize,
        batched=True,
        batch_size=1,
        remove_columns=dataset.column_names,
    )

login()
wandb.login(key="0444d0091009eba5725ae52fd071f747971c78d7")

base_dir = '/content/AideRL'

# Load all the journal files for the teacher GPT-4o model
gpt_dir = os.path.join(base_dir, 'gpt')
competitions = [os.path.basename(x) for x in glob.glob(os.path.join(gpt_dir,'*'))]
gpt_journals = {}
for comp in competitions:
  gpt_journals[comp] = read_json_file(os.path.join(gpt_dir, comp, 'logs', 'journal.json'))

# Read the instructions (only stored in gpt directory)
instructions = {}
for comp in competitions:
  instructions[comp] = read_txt_file(os.path.join(gpt_dir, comp, 'logs', 'instructions.txt'))

# Load all the journal files for the student Llama 3.2 1b model
llama_dir = os.path.join(base_dir, 'llama3.2')
llama_competitions = [os.path.basename(x) for x in glob.glob(os.path.join(llama_dir,'*'))]
llama_journals = {}
for comp in llama_competitions:
  llama_journals[comp] = read_json_file(os.path.join(llama_dir, comp, 'logs', 'journal.json'))

for comp in llama_competitions:
  for node in llama_journals[comp]['nodes']:
    node["model_output"] =  node['plan'] + "\n" + "'''\n" + node['code'] + "'''" # convert journal back to original model output
  for node in gpt_journals[comp]['nodes']:
    node["model_output"] =  node['plan'] + "\n" + "'''\n" + node['code'] + "'''" # convert journal back to original model output

preference_data = {}
for comp in competitions:
  preference_data[comp] = create_preference_dataset(gpt_journals[comp]['nodes'], llama_journals[comp]['nodes'])
  for data in preference_data[comp]:
    data['prompt'] = instructions[comp]

data = []
for comp in preference_data:
  data.extend(preference_data[comp])

# Convert the list of dictionaries to a Hugging Face Dataset
dataset = Dataset.from_list(data)

# Load the pre-trained language model
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
reward_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, trust_remote_code=True, num_labels=1
)

dataset = prepare_dataset(dataset, tokenizer)

# Explicitly create train and eval datasets using slices
eval_dataset = dataset.select(range(100))  # Take first 100 samples for evaluation
train_dataset = dataset.select(range(100, len(dataset)))  # Take the rest for training
print(f"train_dataset: {train_dataset}")
print(f"eval_dataset: {eval_dataset}")

# 3. Create a RewardTrainer instance
# Define your reward configuration
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

training_args = RewardConfig(
    output_dir="/content/reward_model_output",  # Directory to save the model
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,  # Adjust as needed
    learning_rate=1e-5,   # Adjust as needed
    center_rewards_coefficient=0.01,
    remove_unused_columns=False,
    logging_steps=500,
)

trainer = RewardTrainer(
    model=reward_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)

trainer.train()

trainer.save_model("/content/trained_reward_model_llama1B")

