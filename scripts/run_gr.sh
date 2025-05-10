#!/bin/bash

source .env
echo "HF_TOKEN: $HF_TOKEN" | cut -c1-20

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=0,1
# export VLLM_USE_V1=0

# Array of models
MODELS=(
    # "TheFinAI/FinLLaMA-instruct"
    # "TheFinAI/finma-7b-full"

    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Meta-Llama-3-8B-Instruct"
    # "meta-llama/Meta-Llama-3-70B-Instruct"
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "Qwen/Qwen2.5-72B-Instruct"
    # "google/gemma-2-2b-it"
    # "google/gemma-2-9b-it"
    # "google/gemma-2-27b-it"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.1"
    # "mistralai/Mistral-7B-Instruct-v0.3"

    # "ilsp/Meltemi-7B-Instruct-v1.5"
    # "ilsp/Llama-Krikri-8B-Instruct"
    # "TheFinAI/plutus-8B-instruct"

    # "Qwen/QwQ-32B"

    # "gpt-4.5-preview"
    # "o3-mini"
    # "gpt-4o"
    # "gpt-4o-mini"
    # "gpt-4"
    # "gpt-3.5-turbo-0125"

    # Multifinben
    # "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    # "google/gemma-3-4b-it"
    # "google/gemma-3-27b-it"
    # "Qwen/Qwen2.5-Omni-7B"
    "TheFinAI/FinMA-ES-Bilingual" # 4096
    # "Duxiaoman-DI/Llama3.1-XuanYuan-FinX1-Preview"
    # "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
)

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"

    # # 1024
    # lm_eval --model hf \
    #     --model_args "pretrained=$MODEL,max_length=1024" \
    #     --tasks gr \
    #     --num_fewshot 0 \
    #     --device cuda:2 \
    #     --batch_size auto \
    #     --output_path ../results/gr \
    #     --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
    #     --log_samples \
    #     --apply_chat_template \
    #     --include_path ../tasks/plutus

    # # 8192
    # lm_eval --model hf \
    #     --model_args "pretrained=$MODEL,max_length=8192" \
    #     --tasks gr_long \
    #     --num_fewshot 0 \
    #     --device cuda:2 \
    #     --batch_size auto \
    #     --output_path ../results/gr \
    #     --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
    #     --log_samples \
    #     --apply_chat_template \
    #     --include_path ../tasks/plutus

    # 1024
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=2,gpu_memory_utilization=0.8,max_model_len=1024" \
        --tasks gr \
        --batch_size auto \
        --output_path ../results/gr \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/plutus

    # 8192
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=2,gpu_memory_utilization=0.8,max_length=8192" \
        --tasks gr_long \
        --batch_size auto \
        --output_path ../results/gr \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/plutus

    # # api-openai
    # lm_eval --model openai-chat-completions \
    #     --model_args "model=$MODEL, max_tokens=8192" \
    #     --tasks gr_long \
    #     --output_path ../results/gr \
    #     --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
    #     --log_samples \
    #     --apply_chat_template \
    #     --include_path ../tasks/plutus
        
    echo "Finished evaluating model: $MODEL"
done

# # api-deepseek
# lm_eval --model deepseek-chat-completions \
#     --model_args "model=deepseek-chat,max_gen_toks=128,num_concurrent=10" \
#     --tasks GRFINNUM,GRFINTEXT \
#     --output_path ../results/gr \
#     --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
#     --log_samples \
#     --apply_chat_template \
#     --include_path ../tasks/plutus
