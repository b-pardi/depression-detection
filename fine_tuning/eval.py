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
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

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


        del inputs, out, logits, mask
        torch.cuda.empty_cache()
        
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
    
    
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def evaluate_split_chunked(
    model,
    tokenizer,
    ctx,
    ds,
    text_col,
    label_col,
    device,
    max_chunk_tokens=2048,
    overlap_tokens=128,
    view_n_model_predictions=0,
):
    """
    Chunk-based evaluation:
      - Split each conversation into overlapping token chunks.
      - For each chunk, build a prompt: [ctx + chunk + 'Answer:'].
      - Ask the model to output 0/1 per chunk.
      - If any chunk predicts 1 -> final prediction = 1, break early.
      - Otherwise final prediction = 0.
    """

    # What token IDs correspond to labels "0" and "1"
    allowed_tokens = get_label_token_ids(tokenizer)  # your existing helper

    # Build a preamble/suffix around each chunk using your context
    chunk_preamble = (
        ctx.strip()
        + "\n\nYou will now be given a portion of the full conversation transcript. "
          "Based only on this portion, decide whether there is any evidence of depression so far.\n\n"
          "Transcript chunk:\n"
    )
    chunk_suffix = "\n\nAnswer:"

    pre_ids = tokenizer(chunk_preamble, add_special_tokens=False)["input_ids"]
    suf_ids = tokenizer(chunk_suffix, add_special_tokens=False)["input_ids"]

    # How many tokens are left for the transcript portion inside each chunk
    max_transcript_tokens = max_chunk_tokens - len(pre_ids) - len(suf_ids)
    if max_transcript_tokens <= 0:
        raise ValueError(
            f"max_chunk_tokens={max_chunk_tokens} is too small for this ctx/preamble. "
            f"Need > {len(pre_ids) + len(suf_ids)}."
        )

    # Make sure overlap is not bigger than the usable chunk size
    overlap_tokens = min(overlap_tokens, max_transcript_tokens // 2)
    step = max_transcript_tokens - overlap_tokens

    if step <= 0:
        raise ValueError(
            f"Overlap {overlap_tokens} is too large. "
            f"Choose overlap < max_transcript_tokens={max_transcript_tokens}."
        )

    # For optional inspection of some examples
    view_idxs = np.array([-1])
    if view_n_model_predictions > 0:
        rng = np.random.default_rng(seed=42)
        view_idxs = rng.integers(low=0, high=len(ds), size=view_n_model_predictions)

    ytrue, ypred = [], []

    for i, row in enumerate(ds):
        transcript = row[text_col]
        label = int(row[label_col])

        # Tokenize the *transcript only*
        transcript_ids = tokenizer(
            transcript,
            add_special_tokens=False
        )["input_ids"]

        # Sliding window over transcript token IDs
        final_pred = 0  # default = no depression
        n_tokens = len(transcript_ids)

        # For debug printing only once per sample if it's selected
        sample_debug_printed = False

        for start in range(0, n_tokens, step):
            chunk_ids = transcript_ids[start:start + max_transcript_tokens]
            chunk_text = tokenizer.decode(chunk_ids)

            # Build full prompt for this chunk
            prompt = chunk_preamble + chunk_text + chunk_suffix

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_chunk_tokens,
            ).to(device)

            with torch.no_grad():
                out = model(**inputs)
                logits = out.logits[:, -1, :]  # logits for next token (the answer)

                # Mask to only allow label tokens (0/1)
                mask = torch.full_like(logits, float("-inf"))
                mask[:, allowed_tokens] = logits[:, allowed_tokens]

                pred_id = mask.argmax(dim=-1).item()
                yhat_chunk = int(tokenizer.decode(pred_id).strip())

                # Optional debug for some random examples
                if (i in view_idxs) and (not sample_debug_printed):
                    top_k_view = 5
                    _, top_token_ids = torch.topk(logits, top_k_view)
                    top_token_ids = top_token_ids[0].tolist()
                    top_tokens = [tokenizer.decode(tid) for tid in top_token_ids]
                    print(f"[Sample {i}] Raw TOP-{top_k_view} token IDs: {top_token_ids}")
                    print(f"[Sample {i}] Raw TOP-{top_k_view} tokens: {top_tokens}")

                    _, top_token_ids_masked = torch.topk(mask, top_k_view)
                    top_token_ids_masked = top_token_ids_masked[0].tolist()
                    top_tokens_masked = [tokenizer.decode(tid) for tid in top_token_ids_masked]
                    print(f"[Sample {i}] MASKED TOP-{top_k_view} token IDs: {top_token_ids_masked}")
                    print(f"[Sample {i}] MASKED TOP-{top_k_view} tokens: {top_tokens_masked}")
                    print(f"[Sample {i}] True label: {label}")
                    print(f"[Sample {i}] This chunk prediction: {yhat_chunk}")
                    print(f"[Sample {i}] Chunk prompt snippet:\n{prompt[:800]}...\n-------------------------")
                    sample_debug_printed = True

            # free per-chunk tensors
            del inputs, out, logits, mask
            torch.cuda.empty_cache()

            # Early exit: any chunk flagging depression -> final = 1
            if yhat_chunk == 1:
                final_pred = 1
                break

        ytrue.append(label)
        ypred.append(final_pred)

    # Metrics over final per-patient predictions
    acc = accuracy_score(ytrue, ypred)
    macro_f1 = f1_score(ytrue, ypred, average="macro")
    micro_f1 = f1_score(ytrue, ypred, average="micro")
    print("Accuracy:", acc)
    print("Macro-F1:", macro_f1)
    print("Micro-F1:", micro_f1)
    print("\nClassification report:\n", classification_report(ytrue, ypred, digits=4))
    print("\nConfusion matrix:\n", confusion_matrix(ytrue, ypred))

    return ytrue, ypred
