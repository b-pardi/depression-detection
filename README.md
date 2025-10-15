# depression-detection
Do you write code? Do you have depression??? Us too :)

## Setup and Run

1. Download llama models variants and put them in the `models/` folder
These can be found at various huggingface repos such as:

- 3.2: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF
- 3.1: https://huggingface.co/bartowski/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF

2. Change the `model_fn` attr of the Config dataclass found in `configs/path_config.py` to match the llama variant downloaded. The default (and currently only one tested) variant is `Llama-3.2-3B-Instruct-Q8_0.gguf`

3. Classic venv stuff:

```bash
python -m venv venv
.\venv\Scripts\activate     # WINDOWS
source ./venv/bin/activate  # MAC/LINUX
pip install -r requirements.txt
```

4. GPU support (optional): 
To enable GPU support one _may_ need to export an environment variable and force a rebuild of llama_cpp.
**Note:** The reinstall command will take quite a while, if it looks like it froze, just let it keep going for up to 1-2 hrs.
- `$env:CMAKE_ARGS="-DGGML_CUDA=on"` OR Linux: `CMAKE_ARGS="-DGGML_CUDA=on"`
- `pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python`

**Note:** Current methods tested with Python 3.10.6, however it is likely to still work with other versions (e.g. 3.12.x) as well, though not tested.


## Chat Usage
After all initial setup is complete, simply run `python main.py` to begin with the chatbot.
Converse with it as desired, there is currently a barebones job description in `ctx/sys_msg.txt` that you can read to see what it's trying to do.
There are 3 commands that can be entered in the chat at any point:

- `/exit` or `/quit`: closes the chatbot without saving
- `/reset`: wipes the slate clean and removes all chat history (that wasn't saved already) from the current conversation
- `/save`: save the chat log to `output/chat_logs/<config.model_fn-datetime>.jsonl`


## File Structure

depression-detection (root)/
├── configs/                            # contains config dataclasses with global objects of those classes for each param changing/passing
│   ├── chat_model_config.py            # params for configuring the llm chat bot (frontend)
│   ├── path_model_config.py            # paths and filenames
│   ├── retrieval_model_config.py       # params for configuring the RaG llm (backend)
├── ctx/                                # contains all system message text files for llms
│   ├── sys_msg.txt/                    # chat bot context
├── data/
│   ├── stuff/                          
│   │   ├── temp
├── llm_scripts/                        # scripts for llm use/interaction
│   ├── __init__.py             
│   ├── chat_llm.py                     # contains methods and llama_cpp wrapper class for chat bot
├── models/                             # where downloaded llama model variants are placed (ex below)
│   ├── PLACE_LLAMA_MODELS_HERE   
│   ├── Llama-3.2-3B-Instruct-Q8_0.gguf   
│   ├── Meta-Llama-3-8B-Instruct.Q8_0.gguf  
├── output/                             # where llm's skill selections are output. 
│   ├── chat_logs/
│       ├── <model-datetime>/           # output chat logs from the chosen llama model 
├── utils/ 
│   ├── __init__.py             
│   ├── initializers.py                 # setup fns
├── venv/
├── main.py                             # entry point
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt