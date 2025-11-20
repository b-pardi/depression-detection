import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from .preprocessing import  build_prompt

def analyze_prompt_lengths(
    df,
    tokenizer,
    text_col,
    label_col,
    max_length=512,
):
    lengths = []
    too_long = 0

    for text, label in zip(df[text_col], df[label_col]):
        full_prompt = build_prompt(text, label)  # training-time prompt incl label
        ids = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]
        L = len(ids)
        lengths.append(L)
        if L > max_length:
            too_long += 1

    lengths = np.array(lengths)
    stats = {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p95": int(np.percentile(lengths, 95)),
        "p98": int(np.percentile(lengths, 98)),
        "p99": int(np.percentile(lengths, 99)),
        "num_samples": len(lengths),
        "num_too_long": int(too_long),
        "frac_too_long": float(too_long / len(lengths)),
    }
    return stats

def evaluate_split(model, tokenizer, ctx, ds, text_col, label_col, device):
    # token IDs for "0", "1", "2"
    allowed_tokens = [
        tokenizer.encode(str(i), add_special_tokens=False)[0] for i in [0, 1, 2]
    ]

    ytrue, ypred = [], []
    for row in ds:
        # build inference prompt -> no label appended
        text, label = row[text_col], int(row[label_col])
        prompt = build_prompt(ctx, text, label=None)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[:, -1, :]  # logits for next token
            mask = torch.full_like(logits, float("-inf"))
            mask[:, allowed_tokens] = logits[:, allowed_tokens]
            pred_id = mask.argmax(dim=-1).item()
            yhat = int(tokenizer.decode(pred_id).strip())

        ytrue.append(int(label))
        ypred.append(yhat)

    acc = accuracy_score(ytrue, ypred)
    macro_f1 = f1_score(ytrue, ypred, average="macro")
    micro_f1 = f1_score(ytrue, ypred, average="micro")
    print("Accuracy:", acc)
    print("Macro-F1:", macro_f1)
    print("Micro-F1:", micro_f1)
    print("\nClassification report:\n", classification_report(ytrue, ypred, digits=4))
    print("\nConfusion matrix:\n", confusion_matrix(ytrue, ypred))

    return ytrue, ypred