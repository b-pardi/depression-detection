from dataclasses import dataclass, field, fields
from typing import Optional, List, Any, Tuple, Dict
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

    # output
    ckpt_dir_tag = 'init-training'  # str to attach to the end of default log dir name as a personal identifier of the training run
    #log_dir: Path = Path(f"logs/{datetime.now().strftime('%m%d-%H%M%S')}-{ckpt_dir_tag}")
    log_dir: Path = field(init=False)

    # misc
    context_path: Path = Path("ctx/sys_msg.txt") # path to text file containing llm assignment that is used with text samples to create prompts
    seed: int = 42

    # supervised fine tuning config
    tuner_cfg: Dict[str, Any] = field(default_factory=lambda: {
        'max_length': 'auto',
        'per_device_train_batch_size': 2,
        'per_device_eval_batch_size': 2,
        'gradient_accumulation_steps': 8,
        'learning_rate': 2e-4,
        'lr_scheduler_type': "cosine",
        'warmup_ratio': 0.03,
        'num_train_epochs': 5,
        'eval_strategy': "steps",
        'logging_steps': 5,
        'eval_steps': 5,
        'save_steps': 100,
        'bf16': True,
        'gradient_checkpointing': True,
        'save_total_limit': 2,
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

        # ensure dir structure is setup
        make_dirs = [
            self.log_dir,
            self.data_path,
        ]
        for d in make_dirs:
            d.mkdir(exist_ok=True, parents=True)


ft_cfg = FineTuningConfig()