#!/bin/bash

source .env
echo "HF_TOKEN: $HF_TOKEN"

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=3

# Array of models
MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
    "TheFinAI/fl-MED-SYN0-CLEVELAND-merged"
    "TheFinAI/fl-MED-SYN0-HUNGARIAN-train-merged"
    "TheFinAI/fl-MED-SYN0-SWITZERLAND-merged"
    "TheFinAI/fl-MED-SYN0-VA-merged"
    "TheFinAI/fl-SYNC0-merged"
)

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"

    # 1024
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=1,gpu_memory_utilization=0.8,max_model_len=1024" \
        --tasks fl \
        --batch_size auto \
        --output_path ../results \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-fl-0shot,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/federal_learning

    echo "Finished evaluating model: $MODEL"
done