#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --account=PAS3272
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

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

# Reduce fragmentation to avoid OOM during flex_attention block mask allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


cd olmes/oe_eval/dependencies/safety
bash install.sh



dataset_name=(
    "gsm8k"
    "mbpp"
    "ifeval"
    "harmbench::default"
    "xstest::default"

)
model_path=/users/PAS2526/carterglazer/glazer77/CSE-5525-Final/sft-no-olmo-tablegpt

for dataset in "${dataset_name[@]}"; do
    echo "Evaluating on ${dataset}..."

    if [ "${dataset}" = "gsm8k" ]; then
        uv run olmes \
            --model ${model_path} \
            --task ${dataset} \
            --num-shots 8 \
            --model-args '{"chat_model": true}' \
            --output-dir $model_path-eval-${dataset}
    else
        uv run olmes \
            --model ${model_path} \
            --task ${dataset} \
            --model-args '{"chat_model": true}' \
            --output-dir $model_path-eval-${dataset}
    fi
done