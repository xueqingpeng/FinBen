#!/bin/bash

source .env
echo "HF_TOKEN: $HF_TOKEN" | cut -c1-20

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export VLLM_USE_V1=0

# Array of models
MODELS=(
    # Test
    # "meta-llama/Llama-3.2-1B-Instruct"

    # Multifinben
    # "o3-mini"
    # "gpt-4o"
    # "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    "google/gemma-3-4b-it"
    "google/gemma-3-27b-it"
    "Qwen/Qwen2.5-32B-Instruct"
    "Qwen/Qwen2.5-Omni-7B"
    # "TheFinAI/finma-7b-full"
    "Duxiaoman-DI/Llama3.1-XuanYuan-FinX1-Preview"
    "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
    # "TheFinAI/FinMA-ES-Bilingual" # 4096
    "TheFinAI/plutus-8B-instruct"
)

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"

    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_length=8192" \
        --tasks ml \
        --batch_size auto \
        --output_path ../results/ml \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-ml,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/multilingual

    # # api-openai
    # lm_eval --model openai-chat-completions \
    #     --model_args "model=$MODEL, max_tokens=8192" \
    #     --tasks ml \
    #     --output_path ../results/ml \
    #     --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-ml,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
    #     --log_samples \
    #     --apply_chat_template \
    #     --include_path ../tasks/multilingual
        
    echo "Finished evaluating model: $MODEL"
done
