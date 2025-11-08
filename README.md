# depression-detection
Do you write code? Do you have depression??? Us too :)

# Depression Detection via RAG

A clinical decision support system using Retrieval-Augmented Generation (RAG) for depression diagnosis based on DSM-5-TR and ICD-11 diagnostic criteria.

> **Note:** This is an academic research project. Not intended for clinical use.

## Overview

This system combines authoritative clinical materials (DSM-5-TR, ICD-11, APA practice guidelines) with LLMs to demonstrate how RAG can support structured diagnostic assessment for depression and mood disorders.

## Clinical Dataset

Our RAG is built on authoritative clinical materials:

- **DSM-5-TR** (2022) - Primary diagnostic criteria
- **ICD-11** (2024) - International diagnostic standard
- **APA Practice Guidelines** - Major Depressive Disorder (2010), Bipolar Disorder (2010), Depression Across Age Cohorts (2019)
- **SCID-5** (2016) - Structured Clinical Interview
- **WHO Depression Materials** (2025) - Patient-facing descriptions

See `clinical_dataset/metadata.json` for complete citations.

## Evaluation

**Test Data:**
- **DIAC-WOZ** - Clinical dialogue dataset (unseen test data)

**Experiments/models:**
- `clinical_rag.ipynb` - Main RAG system
- `comparison.ipynb` - Model and strategy comparisons
- `social_media.ipynb` - Social media classification model

## Setup

### Prerequisites
- Python 3.10+ (tested with 3.10.6)
- CUDA-capable GPU (optional, recommended)

### Installation

1. Clone and navigate to repository

2. Create virtual environment:
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source ./venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. **(Optional)** Enable GPU support:
```bash
# Windows
$env:CMAKE_ARGS="-DGGML_CUDA=on"

# Mac/Linux
export CMAKE_ARGS="-DGGML_CUDA=on"

# Reinstall with CUDA (takes 1-2 hours)
pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

5. Place GGUF models in `models/` directory

## Usage

**Notebooks:**
```bash
jupyter notebook clinical_rag.ipynb    # Main system
jupyter notebook comparison.ipynb      # Comparisons
jupyter notebook social_media.ipynb    # Social media analysis
```

**CLI:**
```bash
python main.py
```

## File Structure
```bash
depression-detection/
├── clinical_dataset/                   # Clinical materials for RAG
│   ├── metadata.json                   
│   ├── DSM-5-TR.pdf
│   ├── APA_MDD_Guideline.pdf
│   ├── APA_Bipolar_Guideline.pdf
│   ├── APA_Depression_Age_Cohorts.pdf
│   ├── ICD-11_Mental_Behavioral.pdf
│   ├── SCID-5.pdf
│   └── WHO_Depression_Factsheet.pdf
├── configs/                            # Configuration files
│   ├── chat_model_config.py            
│   ├── path_model_config.py            
│   └── retrieval_model_config.py       
├── ctx/                                # System prompts
│   └── sys_msg.txt                     
├── data/
│   └── stuff/                          
│       └── temp/
├── llm_scripts/                        # LLM modules
│   ├── __init__.py             
│   └── chat_llm.py                     
├── models/                             # LLaMA models
│   ├── PLACE_LLAMA_MODELS_HERE   
│   ├── Llama-3.2-3B-Instruct-Q8_0.gguf   
│   └── Meta-Llama-3-8B-Instruct.Q8_0.gguf  
├── output/                             # Generated outputs
│   └── chat_logs/
│       └── <model-datetime>/            
├── utils/ 
│   ├── __init__.py             
│   └── initializers.py                 
├── venv/
├── clinical_rag.ipynb                  # Main RAG evaluation
├── comparison.ipynb                    # Model comparisons
├── social_media.ipynb                  # Social media analysis
├── main.py                             # CLI entry point
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Disclaimers

**This is a research prototype:**
- NOT for clinical diagnosis or patient care
- NOT a substitute for professional evaluation
- Depression diagnosis requires trained clinical professionals
- Outputs are demonstrations of RAG technology only

## References

Complete citations in `clinical_dataset/metadata.json`

Key sources:
- American Psychiatric Association. (2022). *DSM-5-TR*
- World Health Organization. (2024). *ICD-11*
- Fiest et al. (2014). Validated case definitions for depression. *BMC Psychiatry*, 14, 289.

---

**Crisis Resources:**
- **US:** National Suicide Prevention Lifeline: 988
- **US:** Crisis Text Line: Text HOME to 741741
- **International:** https://www.iasp.info/resources/Crisis_Centres/