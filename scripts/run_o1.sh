#!/bin/bash

source .env
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Array of models
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    "meta-llama/Llama-3-70B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct-math"
)

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"

    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.8" \
        --tasks complong \
        --batch_size auto \
        --output_path ../results/o1 \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-o1,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/o1
        
    echo "Finished evaluating model: $MODEL"
done