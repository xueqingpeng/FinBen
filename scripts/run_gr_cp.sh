#!/bin/bash

source .env
echo "HF_TOKEN: $HF_TOKEN"

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=0

# Array of models
MODELS=(
    "TheFinAI/Plutus-base-new"
)

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"

    # 1024
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=1,gpu_memory_utilization=0.8,max_model_len=1024" \
        --tasks gr \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path ../results \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-gr-5shot,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --include_path ../tasks/plutus

    # 8192
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=1,gpu_memory_utilization=0.8,max_length=8192" \
        --tasks gr_long \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path ../results \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-gr-5shot,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --include_path ../tasks/plutus
        
    echo "Finished evaluating model: $MODEL"
done