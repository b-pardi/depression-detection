from dataclasses import dataclass, field, fields
from typing import Optional, List, Any, Tuple, Dict
from pathlib import Path

@dataclass
class Retrieval_Model_Config:
    # initializing llama model
    max_ctx_tokens: int = 1000,
    n_threads: int = 8



retr_cfg = Retrieval_Model_Config()