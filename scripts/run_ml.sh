#!/bin/bash

source .env
echo "HF_TOKEN: $HF_TOKEN" | cut -c1-20

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,expandable_segments:True,garbage_collection_threshold:0.6"
export VLLM_LOG_LEVEL=DEBUG

# Array of models
MODELS=(
    # Test
    # "meta-llama/Llama-3.2-1B-Instruct"

    # Multifinben
    # "o3-mini"
    # "gpt-4o"
    # "gpt-5"
    # "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
    # "google/gemma-3-4b-it"
    # "google/gemma-3-27b-it"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "Qwen/Qwen2.5-Omni-7B"
    # "TheFinAI/finma-7b-full" # 2048 # 
    # "Duxiaoman-DI/Llama3.1-XuanYuan-FinX1-Preview"
    # "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
    # "TheFinAI/FinMA-ES-Bilingual" # 4096
    # "TheFinAI/plutus-8B-instruct"
)

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"

    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_num_seqs=1,max_length=8192,enforce_eager=True" \
        --tasks ml \
        --batch_size auto \
        --output_path ../results/ml \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-ml,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/multilingual \
        # --limit 10

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
