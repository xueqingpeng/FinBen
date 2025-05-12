#!/bin/bash

source .env
echo "HF_TOKEN: $HF_TOKEN"

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=0

# Array of models
MODELS=(
    # Multifinben
    "gpt-4o"
    "o3-mini"
)

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"
        
    # api-openai
    lm_eval --model openai-chat-completions \
        --model_args "model=$MODEL, max_tokens=2048" \
        --tasks dolfin \
        --output_path ../results/dolfin \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-dolfin,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/dolfin
        
    echo "Finished evaluating model: $MODEL"
done