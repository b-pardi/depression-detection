import torch
import pandas as pd
import numpy as np
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from .preprocessing import build_prompt

def load_model(lora_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(lora_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForCausalLM.from_pretrained(lora_dir, device_map=device, dtype=torch.bfloat16)
    model.eval()
    return model, tokenizer

@torch.no_grad()
def predict_label_from_text(model, ctx, tokenizer, text, label_tokens, device):
    prompt = build_prompt(ctx, text)
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    resp = model(**inputs)
    logits = resp.logits[:, -1, :]  # (1, vocab_size)

    mask = torch.full_like(logits, float('-inf'))
    mask[:, label_tokens] = logits[:, label_tokens]

    pred_id = mask.argmax(dim=-1).item()
    return pred_id  # or convert to int(label) / decode