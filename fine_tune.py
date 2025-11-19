import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from configs.fine_tune_config import ft_cfg
from fine_tuning.initialize import *

def main():
    tuner.train()
    tuner.save_model(f'{ft_cfg.log_dir}/final')
    tokenizer.save_pretrained(f'{ft_cfg.log_dir}/final')


#evaluate(df_test[ft_cfg.x_col_name], df_test[ft_cfg.y_col_name])

if __name__ == '__main__':
    main()