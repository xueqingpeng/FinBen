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
    "meta-llama/Llama-4-Scout-17B-16E-Instruct"
)

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"
    
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.95,max_model_len=1024" \
        --tasks dolfin \
        --batch_size auto \
        --output_path ../results/dolfin \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-dolfin,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/dolfin
        
    # # api-openai
    # lm_eval --model openai-chat-completions \
    #     --model_args "model=$MODEL, max_tokens=2048" \
    #     --tasks dolfin \
    #     --output_path ../results/dolfin \
    #     --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-dolfin,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
    #     --log_samples \
    #     --apply_chat_template \
    #     --include_path ../tasks/dolfin
        
    echo "Finished evaluating model: $MODEL"
done