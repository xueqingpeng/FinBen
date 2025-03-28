#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export HF_TOKEN='your hf token here'
export HF_HUB_READ_TIMEOUT=180
export HF_HUB_CONNECT_TIMEOUT=180

set -e

# SHOTS=(
#         0
#       )

# MODELS=(
#     "Qwen/Qwen2.5-1.5B-Instruct"
#     "Qwen/Qwen2.5-0.5B-Instruct"
#     "meta-llama/Llama-3.1-8B-Instruct"
#     )


# Run the Hugging Face VLLM evaluation command
# for MODEL in "${MODELS[@]}"; do
#     echo "running model: $MODEL"
#     for SHOT in "${SHOTS[@]}"; do
#         # lm_eval --model openai-chat-completions \
#         #     --model_args "model=$MODEL" \
#         #     --tasks XBRL_NER \
#         #     --num_fewshot "$SHOT" \
#         #     --output_path results/xbrl_ner \
#         #     --use_cache ./cache1 \
#         #     --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-xbrl-tagging-ner-0-shot-results,push_results_to_hub=False,push_samples_to_hub=False,public_repo=False" \
#         #     --log_samples \
#         #     --apply_chat_template \
#         #     --include_path ./tasks
        
#         lm_eval --model vllm \
#         --model_args "pretrained=$MODEL,tensor_parallel_size=2,gpu_memory_utilization=0.90,max_model_len=2048" \
#         --tasks XBRL_NER \
#         --num_fewshot "$SHOT" \
#         --output_path results/xbrl_ner \
#         --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-xbrl-tagging-ner-0-shot-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
#         --log_samples \
#         --apply_chat_template \
#         --include_path ./tasks
#         # --batch_size 1 \
#         sleep 1
#     done
#     sleep 3
# done


lm_eval --model vllm \
        --model_args "pretrained=google/txgemma-27b-chat,tensor_parallel_size=2,gpu_memory_utilization=0.90,max_model_len=2048" \
        --tasks XBRL_NER \
        --num_fewshot 0 \
        --output_path results/xbrl_ner \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-xbrl-tagging-ner-0-shot-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ./tasks
        # --batch_size 1 \


        
# output message
echo "Evaluation completed successfully!"
