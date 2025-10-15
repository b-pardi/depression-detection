import datetime as dt
import os
from pathlib import Path

from llm_scripts.chat_llm import ChatBot
from configs import path_cfg, chat_cfg
from utils import load_llama

def log_path():
    log_fp = path_cfg.chat_log_dir / path_cfg.model_fn / f"chat-{dt.datetime.now():%Y%m%d-%H%M%S}.jsonl"
    return log_fp

def main():
    # start local server to query llm
    chat_llm = load_llama(path_cfg.model_fp, chat_cfg)
    bot = ChatBot.from_config(chat_llm)
    chat_log_fp = log_path()

    print("\nChat ready. Type your message and press Enter.")
    print("Commands: /reset  (clears history),  /save  (writes transcript),  /exit\n")

    while True: # chat loop
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user:
            continue

        if user.lower() in ("/exit", "/quit"):
            break

        if user.lower() == "/reset":
            bot.reset(system_message=bot.messages[0]["content"] if bot.messages and bot.messages[0]["role"] == "system" else None)
            print("(history cleared)")
            continue

        if user.lower() == "/save":
            bot.save_transcript(chat_log_fp)
            print(f"(saved) {chat_log_fp}")
            continue

        # normal turn
        bot.ask(user, stream=True)
    bot.save_transcript(chat_log_fp)
    print(f"Session saved to: {chat_log_fp}")

if __name__ == '__main__':
    import os, llama_cpp
    print("llama-cpp-python:", llama_cpp.__version__)
    print("GPU offload supported?", llama_cpp.llama_supports_gpu_offload())
    print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
    main()
    

    # path_cfg.base_data_dir / path_cfg.ctx_fn