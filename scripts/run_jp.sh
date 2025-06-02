#!/bin/bash

source .env
echo "HF_TOKEN: $HF_TOKEN" | cut -c1-20

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Array of models
MODELS=(
    # Multifinben
    # "gpt-4o"
    # "o3-mini"
    # "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    "google/gemma-3-27b-it"
    "Qwen/Qwen2.5-32B-Instruct"
    # "Qwen/Qwen2.5-Omni-7B"
    # "Duxiaoman-DI/Llama3.1-XuanYuan-FinX1-Preview"
    # "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
)

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"
    
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.95,max_model_len=1024" \
        --tasks jp \
        --batch_size auto \
        --output_path ../results/jp \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-jp,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/jp
        
    # # api-openai
    # lm_eval --model openai-chat-completions \
    #     --model_args "model=$MODEL, max_tokens=1024" \
    #     --tasks jp_gen \
    #     --output_path ../results/jp \
    #     --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-jp,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
    #     --log_samples \
    #     --apply_chat_template \
    #     --include_path ../tasks/jp
        
    echo "Finished evaluating model: $MODEL"
done