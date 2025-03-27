#!/bin/bash

source .env
echo "HF_USERNAME: $HF_USERNAME"
echo "HF_TOKEN: $HF_TOKEN" | cut -c1-20

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=0,1

# Array of models
MODELS=(
    ""
    ""
    ""
)

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"

    # 1024
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=2,gpu_memory_utilization=0.8,max_model_len=1024" \
        --tasks finben_paper \
        --batch_size auto \
        --output_path ../results \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-fl-0shot,push_results_to_hub=False,push_samples_to_hub=False,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/finben_paper

    echo "Finished evaluating model: $MODEL"
done