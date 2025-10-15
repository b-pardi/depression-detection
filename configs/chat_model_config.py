from dataclasses import dataclass, field, fields
from typing import Optional, List, Any, Tuple, Dict
from pathlib import Path

@dataclass
class Chat_Model_Config:
    # initializing llama chat model
    chat_fmt: str = 'llama-3'
    max_ctx_tokens: int = 8192
    n_threads: int = 8
    n_gpu_layers: int = -1                  # -1 to have all layers of llama model on gpu
    f16_kv: bool = True                     # for mixed precision
    use_mlock: bool = True                  # pin in memory to avoid swap
    offload_kqv: bool = True                # move key query value ops to gpu as well
    verbose: bool = True  
    
    # for querying model
    temperature: float =0.6 
    top_p: float = 0.5 
    top_k: int = 100 
    max_output_tokens: int = 2048 
    stream: bool = True                     # stream=True -> model response types out live as it generates



chat_cfg = Chat_Model_Config()