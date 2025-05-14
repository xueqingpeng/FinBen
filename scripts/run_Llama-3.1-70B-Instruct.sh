#!/bin/bash

source .env
echo "HF_TOKEN: $HF_TOKEN" | cut -c1-20

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Array of models
MODELS=(
    # Multifinben
    "meta-llama/Llama-3.1-70B-Instruct"
)

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"
    
    echo "******************************zh"

    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.95,max_model_len=1024" \
        --tasks zh-classification \
        --batch_size auto \
        --output_path ../results/zh \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-zh,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/chinese
    
    echo "******************************jp"
    
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.95,max_model_len=1024" \
        --tasks jp \
        --batch_size auto \
        --output_path ../results/jp \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-jp,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/jp
    
    echo "******************************es"
    
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.95,max_model_len=1024" \
        --tasks es \
        --batch_size auto \
        --output_path ../results/es \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-es,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/es
    
    echo "******************************el"

    # 1024
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_model_len=1024" \
        --tasks gr \
        --batch_size auto \
        --output_path ../results/gr \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/plutus

    # 8192
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_length=8192" \
        --tasks gr_long \
        --batch_size auto \
        --output_path ../results/gr \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/plutus
    
    echo "******************************bi"
    
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.95,max_model_len=2048" \
        --tasks dolfin \
        --batch_size auto \
        --output_path ../results/dolfin \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-dolfin,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/dolfin
    
    echo "******************************ml"

    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.95,max_length=8192" \
        --tasks ml \
        --batch_size auto \
        --output_path ../results/ml \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-ml,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/multilingual

        
    echo "Finished evaluating model: $MODEL"
done