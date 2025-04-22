#!/bin/bash

source .env
echo "HF_TOKEN: $HF_TOKEN"

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=0

MODEL="meta-llama/Llama-3.1-8B-Instruct" # "meta-llama/Llama-3.2-1B-Instruct"

lm_eval --model vllm \
	--model_args "pretrained=$MODEL,tensor_parallel_size=1,gpu_memory_utilization=0.8,max_model_len=2048" \
        --tasks dolfin \
        --batch_size auto \
        --output_path ../results \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=False,push_samples_to_hub=False,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/dolfin
