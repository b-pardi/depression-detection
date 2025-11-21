import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from .preprocessing import  build_prompt
from .utils import get_label_token_ids

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

def evaluate_split(model, tokenizer, ctx, ds, text_col, label_col, device, view_n_model_predictions=0):
    '''expects samples already formatted from utils.preprocessing.prep_data(mode='eval') '''
    # token IDs for "0", "1", "2"
    view_idxs = np.array([-1])
    if view_n_model_predictions > 0:
        top_k_view = 5
        rng = np.random.default_rng(seed=42)
        view_idxs = rng.integers(low=0, high=(len(ds)), size=view_n_model_predictions)
    
    allowed_tokens = get_label_token_ids(tokenizer)
    ytrue, ypred = [], []
    for i, row in enumerate(ds):
        # build inference prompt -> no label appended
        prompt, label = row[text_col], int(row[label_col])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[:, -1, :]  # logits for next token
            mask = torch.full_like(logits, float("-inf"))
            mask[:, allowed_tokens] = logits[:, allowed_tokens]
            pred_id = mask.argmax(dim=-1).item()
            yhat = int(tokenizer.decode(pred_id).strip())

            if i in view_idxs:
                _, top_token_ids = torch.topk(logits, top_k_view)
                top_token_ids = top_token_ids[0].tolist()
                top_tokens = [tokenizer.decode(tid) for tid in top_token_ids]
                print(f"TOP-{top_k_view} token IDs: {top_token_ids}\nTOP-{top_k_view} tokens:{top_tokens}")

                _, top_token_ids_masked = torch.topk(mask, top_k_view)
                top_token_ids_masked = top_token_ids_masked[0].tolist()
                top_tokens = [tokenizer.decode(tid) for tid in top_token_ids_masked]
                print(f"TOP-{top_k_view} MASKED token IDs: {top_token_ids_masked}\nTOP-{top_k_view} MASKED tokens:{top_tokens}")

                print(f"For the above token choices, the true label is {label}; and the prompt is:\n{prompt}\n-----------------------")


        ytrue.append(label)
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