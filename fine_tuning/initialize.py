import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from configs.fine_tune_config import ft_cfg
from .utils import read_sys_msg, build_prompt, get_max_token_len
from .logger import JSONMetricsLoggerCallback

# load data
# data columns: [pid, text, label]
# pid -> post id; text -> the social media post; label -> 0|1|2 numeric value for level of depression
df_train = pd.read_csv(ft_cfg.train_csv)
df_test = pd.read_csv(ft_cfg.test_csv)

# grab validation data from train set
df_val = df_train.sample(frac=0.1, random_state=42)
df_train.drop(df_val.index, inplace=True)

# add prompts to raw text samples
train_prompts = [build_prompt(text, label) for text, label in zip(df_train[ft_cfg.x_col_name], df_train[ft_cfg.y_col_name])]
val_prompts = [build_prompt(text, label) for text, label in zip(df_val[ft_cfg.x_col_name], df_val[ft_cfg.y_col_name])]
train_ds = Dataset.from_dict({'text': train_prompts}) # trl.SFTTrainer expects dicts wrapping prompt
val_ds = Dataset.from_dict({'text': val_prompts})

print(train_ds['text'])

# llm context (its assignment)
ctx = read_sys_msg(ft_cfg.context_path)

# init tokenizer for llm
print(ft_cfg.base_llm_dir)
tokenizer = AutoTokenizer.from_pretrained(ft_cfg.base_llm_dir, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# determine max token length if set to 'auto'
if ft_cfg.tuner_cfg.get('max_length', None).lower() == 'auto':
    df_all = pd.concat([df_train, df_test])
    include_percent = 0.95
    max_token_len = get_max_token_len(df_all, tokenizer, ft_cfg.x_col_name, ft_cfg.y_col_name, include_percent=include_percent)
    print(f"*** Max token length set to 'auto' in Supervised Fine Tuner Config. Found a length of {max_token_len} tokens sufficient to include {include_percent * 100}% of all samples")
    ft_cfg.tuner_cfg['max_length'] = max_token_len

# init base model
model = AutoModelForCausalLM.from_pretrained(
    ft_cfg.base_llm_dir,
    dtype=torch.bfloat16,
    quantization_config=ft_cfg.bnb_cfg, # QLoRA
    device_map='auto'
)

# define config for PEFT (Parameter Efficient Fine Tuning)
peft_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)

model = get_peft_model(model, peft_cfg) # update model with peft

# define config for SFT (Supervised Fine Tuning)
train_cfg = SFTConfig(**ft_cfg.tuner_cfg) 
tuner = SFTTrainer( # fine tuning trainer obj
    model=model,
    processing_class=tokenizer,
    args=train_cfg,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    callbacks=[JSONMetricsLoggerCallback(f"{ft_cfg.log_dir}/train_metrics.jsonl")] # save logs instead of just printing them
)

__all__ = [
    'train_prompts',
    'val_prompts',
    'ctx',
    'tokenizer',
    'model',
    'peft_cfg',
    'model',
    'tuner',
]