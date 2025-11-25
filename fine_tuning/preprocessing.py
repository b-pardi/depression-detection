import torch
import random
import pandas as pd
import numpy as np
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from datasets import Dataset

def build_prompt(ctx, text, label=None):
    prompt = f"{ctx}Transcript:\n{text}\n\nAnswer:"
    if label is not None: # label included when training
        prompt += " " + str(label)
    return prompt

def tokenize_labels(tokenizer, labels_list):
    label_tokens = []
    for label_int in labels_list:
        label_token = tokenizer.encode(str(label_int), add_special_tokens=False)[0]
        label_tokens.append(label_token)
    return label_tokens

def get_max_token_len(df, ctx, tokenizer, text_col, label_col, include_percent=0.98):
    lens = []
    for text, label in zip(df[text_col], df[label_col]):
        prompt = build_prompt(ctx, text, label)
        tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
        lens.append(len(tokens))
    
    lens = np.array(lens)
    print(np.sort(lens))
    return int(np.percentile(lens, include_percent * 100))

def filter_samples_over_max_len(tokenizer, ctx, df, text_col, label_col, max_token_len=512):
    keep_idxs = []
    for i, (text, label) in enumerate(zip(df[text_col], df[label_col])):
        prompt = build_prompt(ctx, text, label)
        n_tokens = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
        if n_tokens <= max_token_len:
            keep_idxs.append(i)

    filtered_df = df.iloc[keep_idxs].reset_index(drop=True)
    print(f"Removed {len(df) - len(filtered_df)} samples; {len(filtered_df)} samples remaining out of {len(df)} initially.")

    return filtered_df

def oversample_minority(df, text_col, label_col, drop_prob=0.1, excl_words=None, random_state=42, allowed_imbal_factor=1.0):
    # sample with replacement for minority classes
    label_counts = df[label_col].value_counts()
    max_count = int(label_counts.max() * allowed_imbal_factor)

    dfs = [df] # og samples
    for label, count in label_counts.items():
        if count >= max_count: # dont oversample the majority class
            continue

        n_diff = max_count - count 
        df_label = df[df[label_col] == label]            
        df_sampled = df_label.sample(n=n_diff, replace=True, random_state=random_state)
        df_sampled = df_sampled.copy()
        
        # dropout augmentation
        if excl_words:
            df_sampled[text_col] = df_sampled[text_col].apply(
                lambda t: word_dropout(t, excl_words=excl_words, drop_prob=drop_prob)
            )

        dfs.append(df_sampled)
    df_bal = pd.concat(dfs).sample(frac=1.0, random_state=random_state).reset_index(drop=True) # shuffle
    print(f"Counts per label BEFORE minority oversampling: {label_counts}\nCounts per label AFTER minority oversampling: {df_bal[label_col].value_counts()}")
    return df_bal

def word_dropout(text, excl_words, min_tokens=10, drop_prob=0.05):
    """
    Randomly drops ~drop_prob fraction of tokens, but keeps negation words.
    Avoids altering very short texts.
    """
    tokens = text.split()
    if len(tokens) <= min_tokens:
        return text  # too short to safely drop

    kept_tokens = []
    for tok in tokens:
        low = tok.lower().strip(",.!?;:\"'()[]")
        if low in excl_words:
            kept_tokens.append(tok)
            continue

        if random.random() > drop_prob:
            kept_tokens.append(tok)

    if not kept_tokens:
        return text
    return " ".join(kept_tokens)

def prep_data(tokenizer, ft_cfg, mode='train'):
    mode = mode.lower()
    if mode == 'train':
        df = pd.read_csv(ft_cfg.train_csv)
    elif mode == 'eval':
        df = pd.read_csv(ft_cfg.test_csv)
    else:
        raise ValueError("mode must be 'train' or 'eval'")
    
    # determine max token length if set to 'auto'
    max_len = ft_cfg.tuner_cfg.get('max_length', None)
    if isinstance(max_len, str) and max_len == 'auto':
        include_percent = ft_cfg.token_len_percentile
        max_token_len = get_max_token_len(df, ft_cfg.ctx, tokenizer, ft_cfg.x_col_name, ft_cfg.y_col_name, include_percent=include_percent)
        print(f"*** Max token length set to 'auto' in Supervised Fine Tuner Config. Found a length of {max_token_len} tokens sufficient to include {include_percent * 100}% of all samples")
        ft_cfg.tuner_cfg['max_length'] = max_token_len

    # remove samples over the max token length
    df = filter_samples_over_max_len(tokenizer, ft_cfg.ctx, df, ft_cfg.x_col_name, ft_cfg.y_col_name, max_token_len=ft_cfg.tuner_cfg.get('max_length'))

    val_ds = None
    if mode == 'train': # for training data prep, perform minority oversampling with dropout, and sample a validation set
        # oversample the minority classes to attempt class balance
        print(df[ft_cfg.y_col_name].value_counts())
        df = oversample_minority(df, ft_cfg.x_col_name, ft_cfg.y_col_name, excl_words=ft_cfg.keep_words, allowed_imbal_factor=1.0)
        print(df.value_counts())
        # grab validation data from train set
        df_val = df.sample(frac=0.1, random_state=42)
        df.drop(df_val.index, inplace=True)

        # add prompts to raw text samples
        prompts = [build_prompt(ft_cfg.ctx, text, label) for text, label in zip(df[ft_cfg.x_col_name], df[ft_cfg.y_col_name])]
        val_prompts = [build_prompt(ft_cfg.ctx, text, label) for text, label in zip(df_val[ft_cfg.x_col_name], df_val[ft_cfg.y_col_name])]
        ds = Dataset.from_dict({'text': prompts}) # trl.SFTTrainer expects dicts wrapping prompt
        val_ds = Dataset.from_dict({'text': val_prompts})
    
    elif mode == 'eval': # for eval we don't want oversampling, and labels are separate entry of dataset, not added in the prompt
        prompts = [build_prompt(ft_cfg.ctx, text) for text, label in zip(df[ft_cfg.x_col_name], df[ft_cfg.y_col_name])]
        ds = Dataset.from_dict({'text': prompts, 'labels': df[ft_cfg.y_col_name].tolist()}) # trl.SFTTrainer expects dicts wrapping prompt

    print(f"TEMP DEBUG (should be approx the same): {ft_cfg.tuner_cfg.get('max_length')}, {get_max_token_len(df, ft_cfg.ctx, tokenizer, ft_cfg.x_col_name, ft_cfg.y_col_name, include_percent=1)}")
    print(f"Filtered dataframe: {df}, First Dataset entry: {ds[0]}")


    return ds, val_ds

def prep_diac_woz_data(tokenizer, ft_cfg, diac_woz_data):
  # determine max token length if set to 'auto'
    max_len = ft_cfg.tuner_cfg.get('max_length', None)
    if isinstance(max_len, str) and max_len == 'auto':
        include_percent = ft_cfg.token_len_percentile
        max_token_len = get_max_token_len(diac_woz_data, ft_cfg.ctx, tokenizer, ft_cfg.x_col_name, ft_cfg.y_col_name, include_percent=include_percent)
        print(f"*** Max token length set to 'auto' in Supervised Fine Tuner Config. Found a length of {max_token_len} tokens sufficient to include {include_percent * 100}% of all samples")
        ft_cfg.tuner_cfg['max_length'] = max_token_len


def prep_diac_woz_data(tokenizer, ft_cfg, diac_woz_data):
    """
    Prepare DAIC-WOZ data (loaded from pickle) into a HuggingFace Dataset,
    analogous to prep_data(..., mode='eval').

    Expects:
      - ft_cfg.x_col_name: name of the text field in diac_woz_data (e.g. "conversations")
      - ft_cfg.y_col_name: name of the label field (e.g. "mdd_binary")
      - ft_cfg.ctx: prompt context string
    """

    # 1) Normalize diac_woz_data into a pandas DataFrame
    if isinstance(diac_woz_data, pd.DataFrame):
        df = diac_woz_data.copy()
    elif isinstance(diac_woz_data, dict):
        # Only pull the columns we actually need (text + label)
        df = pd.DataFrame({
            ft_cfg.diac_x_col_name: diac_woz_data[ft_cfg.diac_x_col_name],
            ft_cfg.diac_y_col_name: diac_woz_data[ft_cfg.diac_y_col_name],
        })
    else:
        # You can adjust this depending on how your pickle is structured
        raise TypeError(
            f"diac_woz_data must be a dict or DataFrame, got {type(diac_woz_data)}"
        )

    # Make sure there are no missing text/label rows
    df = df.dropna(subset=[ft_cfg.diac_x_col_name, ft_cfg.diac_y_col_name]).reset_index(drop=True)

    # 2) Determine max token length if set to 'auto'
    #max_len = ft_cfg.tuner_cfg.get('max_length', None)
    #if isinstance(max_len, str) and max_len == 'auto':
    #    include_percent = ft_cfg.token_len_percentile
    #    max_token_len = get_max_token_len(
    #        df,
    #        ft_cfg.ctx,
    #        tokenizer,
    #        ft_cfg.diac_x_col_name,
    #        ft_cfg.diac_y_col_name,
    #       include_percent=include_percent
    #    )
    #    
    #    # Giving an upper bound to the max_token_length
    #    max_token_len = min(max_token_len, 2048)  # or 1536 / 1024
    #    
    #    print(
    #        f"*** Max token length set to 'auto' in Supervised Fine Tuner Config. "
    #        f"Found a length of {max_token_len} tokens sufficient to include "
    #       f"{include_percent * 100}% of all samples"
    #    )
    #    ft_cfg.tuner_cfg['max_length'] = max_token_len
    #
    ## 3) Remove samples over the max token length
    #df = filter_samples_over_max_len(
    #    tokenizer,
    #    ft_cfg.ctx,
    #    df,
    #    ft_cfg.diac_x_col_name,
    #    ft_cfg.diac_y_col_name,
    #    max_token_len=ft_cfg.tuner_cfg.get('max_length')
    #)
    #
    ## 4) Build prompts (NO label appended in the prompt, like eval mode)
    #prompts = [
    #    build_prompt(ft_cfg.ctx, text)
    #    for text in df[ft_cfg.diac_x_col_name]
    #]

    # 5) Create HuggingFace Dataset: text + labels
    #ds = Dataset.from_dict({
    #    'text': prompts,
    #    'labels': df[ft_cfg.diac_y_col_name].astype(int).tolist()
    #})
    
    # 5.A) Create HF Dataset with raw text for iterative chunking later
    ds = Dataset.from_dict({
        ft_cfg.diac_x_col_name: df[ft_cfg.diac_x_col_name].tolist(),   # raw transcripts
        ft_cfg.diac_y_col_name: df[ft_cfg.diac_y_col_name].astype(int).tolist(),  # labels
    })

    # 6) Optional debug info (mirroring your existing prep_data debug)
    #print(
    #    f"TEMP DEBUG (should be approx the same): "
    #    f"{ft_cfg.tuner_cfg.get('max_length')}, "
    #    f"{get_max_token_len(df, ft_cfg.ctx, tokenizer, ft_cfg.x_col_name, ft_cfg.y_col_name, include_percent=1)}"
    #)
    print(f"Filtered DAIC-WOZ dataframe (head):\n{df.head()}")
    print(f"First Dataset entry: {ds[0]}")

    return ds
