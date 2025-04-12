#!/bin/bash

source .env
echo "HF_TOKEN: $HF_TOKEN" | cut -c1-20

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=0,1
# export VLLM_IGNORE_FUSION_LAYER=1
export PYTHONWARNINGS="ignore"

# Array of models
MODELS=(
    "Qwen/Qwen2.5-3B"
    "Qwen/Qwen2.5-3B-Instruct"
    "TheFinAI/fl-cleveland-sft-2"
    "TheFinAI/fl-switzerland-sft-2"
    "TheFinAI/fl-hungarian-sft-2"
    "TheFinAI/fl-va-sft-2"
)

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"

    # 1024
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=2,gpu_memory_utilization=0.8,max_model_len=1024" \
        --tasks fl \
        --batch_size auto \
        --output_path ../results/fl \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-fl-0shot,push_results_to_hub=False,push_samples_to_hub=False,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/federated_learning

    echo "Finished evaluating model: $MODEL"
done