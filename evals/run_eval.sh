#!/bin/bash

#This part is need for OSC users
export CC=gcc
export CXX=g++
export TRITON_CACHE_DIR=/fs/scratch/PAS3272/${USER}/triton_cache


export UV_CACHE_DIR=/fs/scratch/PAS3272/${USER}/.cache/uv  #control your uv caches

# Dummy key to prevent import error in safety-eval (WildGuard doesn't actually use it)
export OPENAI_API_KEY="sk-dummy-not-used"

# Disable vLLM V1 multiprocessing so EngineCore runs inline in the spawned subprocess
# rather than forking a grandchild process that loses CUDA visibility on SLURM
export VLLM_ENABLE_V1_MULTIPROCESSING=0


cd olmes/oe_eval/dependencies/safety
bash install.sh



dataset_name=(
    "gsm8k"
    "mbpp"
    "ifeval"
    "harmbench::default"
    "xstest::default"

)
model_path=path/sft-4

for dataset in "${dataset_name[@]}"; do
    safe_dataset="${dataset//::/_}"
    safe_dataset="${safe_dataset//:/_}"
    
    echo "Evaluating on ${dataset}..."

    uv run olmes \
        --model ${model_path} \
        --task ${dataset} \
        --output-dir $model_path-eval-${safe_dataset}

    output_dir="$model_path-eval-${safe_dataset}"
    if [ -d "$output_dir" ]; then
        find "$output_dir" -depth -name '*:*' | while read -r filepath; do
            dir=$(dirname "$filepath")
            base=$(basename "$filepath")
            safe_base="${base//::/_}"
            safe_base="${safe_base//:/_}"
            mv "$filepath" "$dir/$safe_base"
        done
    fi
done