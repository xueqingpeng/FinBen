#!/bin/bash

source .env
echo "HF_USERNAME: $HF_USERNAME"
echo "HF_TOKEN: $HF_TOKEN" | cut -c1-20

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_READ_TIMEOUT=180
export HF_HUB_CONNECT_TIMEOUT=180

# Array of models
MODELS=(
    "0"
    # "TheFinAI/FinLLaMA-instruct"
    # "TheFinAI/finma-7b-full"
)

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"

    lm_eval --model hf \
        --model_args "pretrained=TheFinAI/finma-7b-full,max_length=2048" \
        --tasks XBRL_NER \
        --batch_size auto \
        --num_fewshot 0 \
        --output_path ../results/xbrl_ner \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-xbrl-tagging-ner-0-shot-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --include_path ../tasks/XBRLtagging
        # --device cuda:0 \
        # --apply_chat_template \

    lm_eval --model vllm \
        --model_args "pretrained=TheFinAI/FinLLaMA-instruct,tensor_parallel_size=1,gpu_memory_utilization=0.90,max_model_len=2048" \
        --tasks XBRL_NER \
        --batch_size auto \
        --num_fewshot 0 \
        --output_path ../results/xbrl_ner \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-xbrl-tagging-ner-0-shot-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/XBRLtagging

#     lm_eval --model openai-chat-completions \
#         --model_args "model=$MODEL" \
#         --tasks XBRL_NER \
#         --batch_size auto \
#         --num_fewshot 0 \
#         --output_path ../results/xbrl_ner \
#         --use_cache ./cache1 \
#         --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-xbrl-tagging-ner-0-shot-results,push_results_to_hub=False,push_samples_to_hub=False,public_repo=False" \
#         --log_samples \
#         --apply_chat_template \
#         --include_path ../tasks/XBRLtagging

    echo "Finished evaluating model: $MODEL"
done
