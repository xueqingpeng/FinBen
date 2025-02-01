#!/bin/bash

export HF_TOKEN="your_hf_token"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=4,5,6,7

# lm_eval --model hf \
#     --model_args "pretrained=ilsp/Meltemi-7B-Instruct-v1.5" \
#     --tasks gr_ner \
#     --num_fewshot 0 \
#     --device cuda:2 \
#     --batch_size 8 \
#     --output_path ../results \
#     --hf_hub_log_args "hub_results_org=TheFinAI,results_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
#     --log_samples \
#     --apply_chat_template \
#     --include_path ../tasks

# Array of model names
MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Meta-Llama-3-70B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-32B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "google/gemma-2-2b-it"
    "google/gemma-2-9b-it"
    "google/gemma-2-27b-it"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "ilsp/Meltemi-7B-Instruct-v1.5"
    "TheFinAI/FinLLaMA-instruct"
)

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"

    # 1024
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_model_len=1024" \
        --tasks gr \
        --batch_size auto \
        --output_path ../results \
        --hf_hub_log_args "hub_results_org=TheFinAI,results_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/plutus

    # 8192
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_length=8192" \
        --tasks gr_long \
        --batch_size auto \
        --output_path ../results \
        --hf_hub_log_args "hub_results_org=TheFinAI,results_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/plutus
    
    echo "Finished evaluating model: $MODEL"
done
