from llama_cpp import Llama
import os
def load_llama(model_path, cfg):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ERROR: Model not found in location specified in the config file: {model_path}")
    
    llm = Llama(
        model_path=str(model_path),
        chat_format=cfg.chat_fmt,
        n_ctx=cfg.max_ctx_tokens,
        n_threads=cfg.n_threads,
        n_gpu_layers=cfg.n_gpu_layers,
        f16_kv=cfg.f16_kv, 
        use_mlock=cfg.use_mlock,
        offload_kqv=cfg.offload_kqv,
        verbose=cfg.verbose
    )

    return llm

def _read_sys_msg(fp):
    with open(fp, 'r', encoding='utf-8') as ctx_file:
        sys_msg = ctx_file.read().strip()
    
    return sys_msg