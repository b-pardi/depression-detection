# Depression Detection via RAG
Do you write code? Do you have depression??? Us too :)


> **Note:** This is an academic research project. Not intended for clinical use.

 ## Overview

Depression detection in conversational AI has traditionally relied on models fine-tuned with social media data. However, this approach may lack clinical grounding needed for reliable mental health applications. 

This project compares three LLM approaches for detecting depression in dialogue:
1. **Fine-tuned models** trained on social media data
2. **RAG-enhanced models** using authoritative clinical materials (DSM-5-TR, ICD-11, APA guidelines)
3. **Prompt-engineered models** with zero additional training

We evaluate all models on **DIAC-WOZ**, a chatbot dialogue dataset with depression labels derived from PHQ-8 clinical assessments. This provides a rigorous test of whether clinical grounding via RAG can match or exceed the performance of domain-specific fine-tuning on real conversational data with validated clinical outcomes.

 ## Evaluation

**Unseen Test Data:**
- **DIAC-WOZ** - Clinical dialogue dataset can be found here: https://dcapswoz.ict.usc.edu/
- We would like to grow this list to create a benchmark for depression detection in LLMs


## Setup

### Prerequisites
- Python 3.13+ (tested with 3.13.9)
- CUDA-capable GPU (optional, recommended)

### Installation

1. Clone and navigate to repository

2. Create virtual environment (conda recommended):
```bash
conda create --name myenv python=3.13
conda activate myenv
```

3. Install dependencies:
```bash
conda install --file requirements.txt
```

## RAG

**Clinical database:**
Our RAG is built on authoritative clinical materials:

- **DSM-5-TR** (2022) - Primary diagnostic criteria
- **ICD-11** (2024) - International diagnostic standard
- **APA Practice Guidelines** - Major Depressive Disorder (2010), Bipolar Disorder (2010), Depression Across Age Cohorts (2019)
- **SCID-5** (2016) - Structured Clinical Interview
- **WHO Depression Materials** (2025) - Patient-facing descriptions

To run the RAG in `clinical_rag.ipynb`, you would need to create a database of clinical criteria including documents such as the DSM-5-TR. The `data_processing_scripts/create_embeddings.ipynb` notebook can be used to walk you through the embedding process.

### Fine Tuning on Social Media Data
1. Request access for and download Llama models from the [Official Meta Llama Downloads Website](https://www.llama.com/llama-downloads/). **Note:** the Llama model used for this project specifically is Llama-3.2-3B-Instruct, and further instructions pertain to that version.

2. Convert Llama weights to hugging face format.
```
python /fine_tuning/convert_llama_weights_to_hf.py    --input_dir "/path/to/llama/weights/folder"   --model_size 3B   --output_dir "/path/to/desired/converted/weights/folder"   --llama_version 3.2   --safe_serialization

# EXAMPLE
python /fine_tuning/convert_llama_weights_to_hf.py    --input_dir "/mnt/c/Users/BPardi/.llama/checkpoints/Llama3.2-3B-Instruct"   --model_size 3B   --output_dir "/mnt/c/Users/BPardi/.llama/hf/Llama-3.2-3B-Instruct"   --llama_version 3.2   --safe_serialization
```

**Note:** The transformers library supposedly ships with this script, but it didn't for me so I included it in this repo. Additionally there were several issues with this script pertaining to deprecated args that I had to remedy.

3. [Download datasets train.csv and test.csv](https://github.com/rafalposwiata/depression-detection-lt-edi-2022/tree/main/data/preprocessed_dataset) and place them in the `data/social_media_posts_folder`

4. Update `configs/fine_tune_config.py` as desired. The only parameter that _needs_ to be updated is the `FineTuningConfig.base_llm_dir` from step 1.

5. Run `python fine_tune.py`

Tips:
- If your GPU VRAM and utilization are near maximum and training appears stalled for a few minutes, decrease `tuner_cfg['per_device_train_batch_size]` and `tuner_cfg['per_device_eval_batch_size]`. Increase `tuner_cfg['gradient_accumulation_steps]` by the same factor the batch sizes were reduced to maintain the same number of gradient steps per epoch.

## File Structure
```bash
depression-detection/
.
.
.
├── fine_tuning/                        # Scripts for fine tuning on social media posts
│   ├── __init__.py             
│   └── chat_llm.py                      
├── logs/                               # Training/Evaluation outputs                         
├── clinical_rag.ipynb                  # Main RAG evaluation
├── fine_tune.py                        # CLI entry point for running fine tuning on social media posts
├── .gitignore
├── LICENSE
├── prompt_engineering.ipynb            # simple prompt engineering approach
├── README.md
└── requirements.txt
```

## Disclaimers

**This is a research prototype:**
- NOT for clinical diagnosis or patient care
- NOT a substitute for professional evaluation
- Depression diagnosis requires trained clinical professionals

**Crisis Resources:**
- **US:** National Suicide Prevention Lifeline: 988
- **US:** Crisis Text Line: Text HOME to 741741
- **International:** https://www.iasp.info/resources/Crisis_Centres/