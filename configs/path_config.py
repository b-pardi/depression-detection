from dataclasses import dataclass, field, fields
from typing import Optional, List, Any, Tuple, Dict
from pathlib import Path
import os

@dataclass
class Path_Config:
    # TODO: CLI args for these?
    model_fn: str = "Llama-3.2-3B-Instruct-Q8_0.gguf"    # llama model of choice (in the `models/` folder)
    model_dir: Path = Path(os.getcwd(), "models/")          # model file path
    model_fp: Path = model_dir / model_fn                   # path to text files w/ proj descriptions
    
    ctx_fp: Path = Path("ctx", "sys_msg.txt")               # name of context file for llm
    
    base_data_dir: Path = Path(os.getcwd(), "data/")        # where all relevant training/tuning data is stored
    out_dir: Path = Path(os.getcwd(), "output/")            # path to output llm generations
    chat_log_dir: Path = Path(out_dir, 'chat_logs')

path_cfg = Path_Config()