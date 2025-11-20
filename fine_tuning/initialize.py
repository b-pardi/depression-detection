import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from configs.fine_tune_config import ft_cfg
from .preprocessing import prep_data
from .logger import JSONMetricsLoggerCallback

# init tokenizer for llm
print(ft_cfg.base_llm_dir)
tokenizer = AutoTokenizer.from_pretrained(ft_cfg.base_llm_dir, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# load data and preprocess training data
# data columns: [pid, text, label]
# pid -> post id; text -> the social media post; label -> 0|1|2 numeric value for level of depression
train_ds, val_ds = prep_data(tokenizer, ft_cfg, mode='train')

# ensure log dir is made and available
ft_cfg.log_dir.mkdir(exist_ok=True, parents=True)

# init base model
model = AutoModelForCausalLM.from_pretrained(
    ft_cfg.base_llm_dir,
    dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(**ft_cfg.bnb_cfg), # QLoRA
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
    'tokenizer',
    'model',
    'peft_cfg',
    'model',
    'tuner',
]