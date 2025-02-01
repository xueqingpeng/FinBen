#!/bin/bash

export HF_TOKEN="your_hf_token"
export OPENAI_API_KEY="your_openai_api_key"
export DEEPSEEK_API_KEY="your_deepseek_api_key"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=2

# Array of model names
MODELS=(
    # "TheFinAI/FinLLaMA-instruct"

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
    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "ilsp/Meltemi-7B-Instruct-v1.5"

    # "gpt-4"
    # "gpt-4o"
    # "gpt-4o-mini"
    # "gpt-3.5-turbo-0125"
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
    #     --output_path ../results \
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
    #     --output_path ../results \
    #     --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
    #     --log_samples \
    #     --apply_chat_template \
    #     --include_path ../tasks/plutus

    # # 1024
    # lm_eval --model vllm \
    #     --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_model_len=1024" \
    #     --tasks gr \
    #     --batch_size auto \
    #     --output_path ../results \
    #     --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
    #     --log_samples \
    #     --apply_chat_template \
    #     --include_path ../tasks/plutus

    # # 8192
    # lm_eval --model vllm \
    #     --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_length=8192" \
    #     --tasks gr_long \
    #     --batch_size auto \
    #     --output_path ../results \
    #     --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
    #     --log_samples \
    #     --apply_chat_template \
    #     --include_path ../tasks/plutus

    # # api-openai
    # lm_eval --model openai-chat-completions \
    #     --model_args "model=$MODEL" \
    #     --tasks GRFINNUM,GRFINTEXT \
    #     --output_path ../results \
    #     --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
    #     --log_samples \
    #     --apply_chat_template \
    #     --include_path ../tasks/plutus
        
    echo "Finished evaluating model: $MODEL"
done

# api-deepseek
lm_eval --model deepseek-chat-completions \
    --model_args "model=deepseek-chat,max_gen_toks=128,num_concurrent=10" \
    --tasks GRFINNUM,GRFINTEXT \
    --output_path ../results \
    --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
    --log_samples \
    --apply_chat_template \
    --include_path ../tasks/plutus
