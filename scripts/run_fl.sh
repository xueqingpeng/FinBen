#!/bin/bash

source .env
echo "HF_TOKEN: $HF_TOKEN" | cut -c1-20

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=0,1
# export VLLM_IGNORE_FUSION_LAYER=1
export PYTHONWARNINGS="ignore"

# Array of models
MODELS=(
    # "Qwen/Qwen2.5-3B" # apply_chat_template
    # "Qwen/Qwen2.5-3B-Instruct"
    
    # "TheFinAI/fl-cleveland-sft-0"
    # "TheFinAI/fl-hungarian-sft-0"
    # "TheFinAI/fl-switzerland-sft-0"
    # "TheFinAI/fl-va-sft-0"
    
    # "TheFinAI/fl-cleveland-sft-1"
    # "TheFinAI/fl-hungarian-sft-1"
    # "TheFinAI/fl-switzerland-sft-1"
    # "ShawnXiaoyuWang/FedMerged-5-5-2025"
    
    "TheFinAI/fl-cleveland-sft-2-adapter"
    "TheFinAI/fl-hungarian-sft-2-adapter"
    "TheFinAI/fl-switzerland-sft-2-adapter"
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
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-fl-0shot,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --include_path ../tasks/federated_learning \
        --apply_chat_template

    echo "Finished evaluating model: $MODEL"
done