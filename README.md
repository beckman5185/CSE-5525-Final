# CSE 5525: Default Project - Spring 2026
## Kathleen Duffey, Carter Glazer, Audrey Beckman

## Overview
LLMs are becoming increasingly important for the modern computer scientist to be able to understand and use. To do this, it is imperative that we know how these models train and work internally - which is the aim of these assignments. Our goal for this project is to build a correct training pipeline by using a ˜1B parameter open LLM provided on Tinker.

## Project Structure

```
├── README.md                 # This file
├── train_sft.py              # Template for Supervised Fine-Tuning
├── train_pref.py             # Template for Preference Optimization
├── configs/                  # Configuration files for training
├── scripts/                  # Utility scripts
└── evals/                    # Evaluation suite (OLMES)
   ├── run_eval.sh           # Script to run evaluations
   └── olmes/                # AI2's Open Language Model Evaluation System
```

Model
```
Llama-3.2-1B Model
```
## How to Run SFT

### 1. Environment Setup

#### 1. Activate enviroment & setup

# Create and activate enviroment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (adjust based on your requirements)
uv pip install tinker

#### 2. Set Tinker API key
```bash
$env:TINKER_API_KEY="your_key_here"
```
#### 3. Run Training with Desired Parameters
```bash
python train_sft.py lora_rank=16 num_epochs=2 batch_size=256 save_every=200 learning_rate=3e-4 lr_schedule=cosine max_length=8096 log_path=runs/sft-2 wandb_project=CSE-5525-Final
```
### 2. Testing
#### 1. Switch to OSC and Request GPU, Then Activate Environment
Following guidelines on the official OSC website and earlier in this readme

#### 2. Download Tinker Weights
```bash
tinker checkpoint download tinker://d3b79b07-d58a-56f2-bc31-a78b6f6dde37:train:0/sampler_weights/final # Or whatever the path is to the sampler weights
```
#### 3. Run transform.py
First, change the arguments in the file to match the file you just created
```bash
python transform.py
```

#### 4. Switch tokenizer_config.json tokenizer
In file you just created, go to the tokenizer: line. Change this from TokenizerBackend to PreTrainedTokenizerFast.

#### 5. Run evaluations
Run evaluations according to olmes documentation provided!

#### Preference Optimization (PREF)
Implement your preference optimization (e.g., DPO, IPO) in `train_pref.py`. This aligns the model using preference data.