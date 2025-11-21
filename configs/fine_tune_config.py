from dataclasses import dataclass, field, fields
from typing import Optional, List, Any, Tuple, Dict, Set
from pathlib import Path
from datetime import datetime
import torch

@dataclass
class FineTuningConfig:
    # base model components
    base_llm_dir: Path = Path("/mnt/c/users/BPardi/.llama/hf/Llama3.2-3B-Instruct")
    base_llm_ckpt_path: Path = base_llm_dir / "consolidated.00.pth"
    base_llm_params_path: Path = base_llm_dir / "params.json"
    base_llm_tokenizer_path: Path = base_llm_dir / "tokenizer.model"

    # data
    data_path: Path = Path("data/social_media_posts/")   # Path to data containing posts + depression level
    train_csv: Path = data_path / "train.csv"
    test_csv: Path = data_path / "test.csv"
    id_col_name: str = 'pid'
    x_col_name: str = 'text'
    y_col_name: str = 'labels'

    # for minority oversampling, these are words to not remove
    keep_words: set[str] = field(default_factory=lambda:{
        "not", "no", "never", "don't", "doesn't", "didn't", "isn't",
        "wasn't", "aren't", "won't", "can't", "angry", "kill",
        "hurt", "anger", "sad", "sadness", "depressed", "depression"
    })
    minority_sampling_drop_rate: float = 0.1

    # output
    ckpt_dir_tag = 'lowerLR-fixedPrompts-smot-smallerbsize'  # str to attach to the end of default log dir name as a personal identifier of the training run
    #log_dir: Path = Path(f"logs/{datetime.now().strftime('%m%d-%H%M%S')}-{ckpt_dir_tag}")
    log_dir: Path = field(init=False)

    # misc
    ctx: str = field(init=False) # llm context (its assignment)
    context_path: Path = Path("ctx/sys_msg.txt") # path to text file containing llm assignment that is used with text samples to create prompts
    device: str = field(init=False)
    seed: int = 42
    token_len_percentile: float = 0.99

    # supervised fine tuning config
    tuner_cfg: Dict[str, Any] = field(default_factory=lambda: {
        'max_length': 'auto',
        'per_device_train_batch_size': 2,
        'per_device_eval_batch_size': 2,
        'gradient_accumulation_steps': 64,
        'learning_rate': 4e-5,
        'lr_scheduler_type': "cosine",
        'warmup_ratio': 0.03,
        'num_train_epochs': 20,
        'eval_strategy': "steps",
        'logging_steps': 5,
        'eval_steps': 25,
        'save_steps': 25,
        'bf16': True,
        'gradient_checkpointing': True,
        'save_total_limit': None,
        'report_to': "none",
        'dataset_text_field': 'text'
    })

    # QLoRA config
    bnb_cfg: Dict[str, Any] = field(default_factory=lambda: {
        'load_in_4bit': True,                   # enables 4-bit quantization
        'bnb_4bit_compute_dtype': torch.bfloat16,
        'bnb_4bit_quant_type': 'nf4',           # nf4 = best QLoRA quantization
        'bnb_4bit_use_double_quant': True       # double-quant improves stability
    })

    def __post_init__(self):
        self.log_dir: Path = Path(f"logs/sft/{datetime.now().strftime('%m%d-%H%M%S')}-{self.ckpt_dir_tag}")
        self.tuner_cfg['output_dir'] = self.log_dir

        # set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # grab context text
        with open(self.context_path, 'r', encoding='utf-8') as ctx_file:
            self.ctx = ctx_file.read().strip()

ft_cfg = FineTuningConfig()