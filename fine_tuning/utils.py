import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def read_sys_msg(fp):
    with open(fp, 'r', encoding='utf-8') as ctx_file:
        sys_msg = ctx_file.read().strip()

def build_prompt(ctx, text, label=None, for_training=True):
    prompt = f"{ctx}Post:\n{text}\n\nAnswer:"
    if label is not None: # label included when training
        prompt += " " + str(label)
    return prompt

def tokenize_labels(tokenizer, labels_list):
    label_tokens = []
    for label_int in labels_list:
        label_token = tokenizer.encode(str(label_int), add_special_tokens=False)[0]
        label_tokens.append(label_token)

def get_max_token_len(df, tokenizer, text_col, label_col, include_percent=0.98):
    lens = []
    for text, label in zip(df[text_col], df[label_col]):
        prompt = build_prompt(text, label)
        tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
        lens.append(len(tokens))
    
    lens = np.array(lens)
    print(np.sort(lens))
    return int(np.percentile(lens, include_percent * 100))

def evaluate(x_prompts, y_labels):
    y_true, y_pred = [], []
    for post, label in zip(x_prompts, y_labels):
        y_pred.append(predict_label_from_text(post))
        y_true.append(label)

    print("Macro-F1:", f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred, digits=4))
    print(confusion_matrix(y_true, y_pred))

@torch.no_grad()
def predict_label_from_text(model, tokenizer, text, label_tokens):
    prompt = build_prompt(text)
    inputs = tokenizer(prompt, return_tensors='pt').to_device() # tokenize prompt
    resp = model(**inputs) # feed tokenized prompt to model
    logits = resp.logits[:, -1, :]

    # mask out output tokens that aren't labels (since that's what it should be outputting)
    # model should be outputting a '0', '1' or '2', so this sets the prob of all tokens outside of those to -inf
    mask = torch.full_like(logits, float('-inf'))
    mask[:, label_tokens] = logits[: label_tokens]