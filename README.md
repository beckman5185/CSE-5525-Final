# CSE 5525: Default Project - Spring 2026
## Kathleen Duffey, Carter Glazer, Audrey Beckman

## Overview
LLMs are becoming increasingly important for the modern computer scientist to be able to understand and use. To do this, it is imperative that we know how these models train and work internally - which is the aim of these assignments. Our goal for this project is to build a correct training pipeline by using a ˜1B parameter open LLM provided on Tinker.

## Model Description
For this model, the training data that we used was the default Tulu 3/OLMo 2 SFT Dataset for SFT and OLMo 2 1B Preference Dataset for DPO/IPO. As the DPO/IPO models were performed from an SFT baseline trained at rank 8, our best SFT model (at rank 16) is the most performant. This shows improvement from the baseline model in terms of instruction-following ability and complex tasks (such as code and math reasoning), but does not improve significantly in the area of safety. Despite the improvements made on the base model, the model remains somewhat limited. Its ability to correctly follow complex instructions or reason in complex ways remains limited (as demonstrated by the low benchmark scores), meaning that this model is more suited for use on simple tasks.

## Project Structure

```
├── README.md                 # This file
├── train_sft.py              # SFT implementation (hand implemented)
├── train_sft_library.py      # Version of SFT implementation - using Tinker cookbook
├── train_pref.py             # DPO implementation
├── train_ipo.py              # IPO implementation
├── train.py                  # Utility script for SFT
├── chat_datasets.py          # Utility script for SFT, DPO, and IPO
├── plot_metrics.py           # Utility script for plotting DPO and IPO metrics
├── transform.py              # Utility script for preparing model for evals
├── runs/                     # All outputs from train_sft.py and train_pref.py
├── output/                   # All merged weights matrices and eval results for SFT
├── dpo_output/               # All merged weights matrices and eval results for DPO (SFT rank 8 and DPO rank 8)
├── exploration1/             # All merged weights matrices and eval results for exploration 1 (data filtering)
├── exploration2/             # All merged weights matrices and eval results for exploration 2 (IPO)
├── tinker_downloads/         # All downloaded checkpoints weights used in evals
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

#### A. Activate enviroment & setup

##### Create and activate enviroment
python -m venv .venv
source .venv/bin/activate

##### Install dependencies (adjust based on your requirements)
uv pip install tinker
pip install peft

#### B. Set Tinker API key
```bash
$env:TINKER_API_KEY="your_key_here"
```
#### C. Run Training with Desired Parameters
```bash
python train_sft.py lora_rank=16 num_epochs=2 batch_size=256 save_every=200 learning_rate=3e-4 lr_schedule=cosine max_length=8096 log_path=runs/sft-2 wandb_project=CSE-5525-Final
```
### 2. Testing
#### A. Switch to OSC and Request GPU, Then Activate Environment
Following guidelines on the official OSC website and earlier in this readme

#### B. Download Tinker Weights
```bash
tinker checkpoint download tinker://d3b79b07-d58a-56f2-bc31-a78b6f6dde37:train:0/sampler_weights/final # Or whatever the path is to the sampler weights
```
#### C. Run transform.py
First, change the arguments in the file to match the file you just created
```bash
python transform.py
```

#### D. Switch tokenizer_config.json tokenizer
In file you just created, go to the tokenizer: line. Change this from TokenizerBackend to PreTrainedTokenizerFast.

#### E. Run evaluations
Run evaluations according to olmes documentation provided!





## How to Run Preference Optimization (DPO)
### 1. Environment Setup

#### A. Activate enviroment & setup
Follow the same instructions from running SFT.

#### B. Set Tinker API key
Follow the same instructions from running SFT.

#### C. Provide Path of SFT Run (Manually)
The path of the SFT baseline to start DPO from was not expected to change between runs, so has been hard-coded into the file. If you would like to run the preference optimization from a different SFT baseline than the rank 8 SFT, change line 256 in train_pref.py. 

```python
   sft_log_path = "runs/sft-rank8"
```

#### D. Run Training with Desired Parameters
The parameters used for the one run of DPO are hard-coded in, but can be changed using the command line as desired (similarly to SFT). The loss type is assumed to be DPO. 

```bash
python train_pref.py
```
### 2. Testing
Follow the same instructions from testing SFT. 


## How to Run Preference Optimization (IPO)
To run IPO, follow the same instructions as DPO, but add a loss_type ipo command line argument when running train_pref.py

```bash
python train_pref.py. loss_type=ipo
```
