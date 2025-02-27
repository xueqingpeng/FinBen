#!/bin/bash

source .env
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=6,7

TASKS=(
        "PassageRank1"
        "PassageRank2"
        "PassageRank3"
        "PassageRank4"
        "PassageRank5"
        "PassageRank6"
        "PassageRank7"
      )

# Run the Hugging Face VLLM evaluation command
for TASK in "${TASKS[@]}"; do
    echo "running task: $TASK"
    
    lm_eval --model vllm \
      --model_args "pretrained=google/gemma-2-27b-it,tensor_parallel_size=2,gpu_memory_utilization=0.90,max_model_len=512" \
      --tasks "$TASK" \
      --num_fewshot 0 \
      --output_path ../results/rank \
      --hf_hub_log_args "hub_results_org=YanAdjeNole,details_repo_name=lm-eval-ranking-0shot-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
      --log_samples \
      --apply_chat_template \
      --include_path ../tasks/passage_rank
      # --batch_size 1 \
    sleep 2
done