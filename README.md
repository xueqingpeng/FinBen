## Setting Up the Environment

1. Navigate to the evaluation folder:
   ```bash
   cd FinBen/finlm_eval/
   ```
2. Create and activate a new conda environment:
   ```bash
   conda create -n finben python=3.12
   conda activate finben
   ```
3. Install the required dependencies:
   ```bash
   pip install -e .
   pip install -e .[vllm]
   ```

## Logging into Hugging Face

Set the Hugging Face token as an environment variable:
   ```bash
   export HF_TOKEN="your_hf_token"
   ```

## Model Evaluation

1. Navigate to the FinBen directory:
   ```bash
   cd FinBen/
   ```

2. Set the VLLM worker multiprocessing method:
   ```bash
   export VLLM_WORKER_MULTIPROC_METHOD="spawn"
   ```

4. Run evaluation:
   ##### Important Notes on Evaluation
      - **0-shot setting:** Use `lm-eval-results` as the results repository.
      - **5-shot setting:** Use `lm-eval-results-gr-5shot` as the results repository.

   ##### GRMultifin
      ```bash
      # 0-shot
      lm_eval --model vllm \
         --model_args "pretrained=meta-llama/Llama-3.2-1B-Instruct,gpu_memory_utilization=0.8,max_model_len=1024"" \
         --tasks GRMultifin \
         --num_fewshot 0 \
         --device cuda:0 \
         --batch_size 8 \
         --output_path results \
         --hf_hub_log_args "hub_results_org=TheFinAI,results_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
         --log_samples \
         --apply_chat_template \
         --include_path ./tasks
   
      # 5-shot
      lm_eval --model vllm \
         --model_args "pretrained=meta-llama/Llama-3.2-1B-Instruct,gpu_memory_utilization=0.8,max_model_len=1024"" \
         --tasks GRMultifin \
         --num_fewshot 5 \
         --device cuda:0 \
         --batch_size 8 \
         --output_path results \
         --hf_hub_log_args "hub_results_org=TheFinAI,results_repo_name=lm-eval-results-gr-5shot,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
         --log_samples \
         --apply_chat_template \
         --include_path ./tasks
      ```
      The accuracy achieved for this command is as follows: 0.3704 for 0-shot and 0.3889 for 5-shot.
   
   ##### GRQA
      ```bash
      # 0-shot
      lm_eval --model vllm \
         --model_args "pretrained=google/gemma-2-27b-it,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_model_len=1024" \
         --tasks GRQA \
         --num_fewshot 0 \
         --batch_size auto \
         --output_path results \
         --hf_hub_log_args "hub_results_org=TheFinAI,results_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
         --log_samples \
         --apply_chat_template \
         --include_path ./tasks
   
      # 5-shot
      lm_eval --model vllm \
         --model_args "pretrained=google/gemma-2-27b-it,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_model_len=1024" \
         --tasks GRQA \
         --num_fewshot 5 \
         --batch_size auto \
         --output_path results \
         --hf_hub_log_args "hub_results_org=TheFinAI,results_repo_name=lm-eval-results-gr-5shot,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
         --log_samples \
         --apply_chat_template \
         --include_path ./tasks
      ```
   
   ##### GRFNS2023
      ```bash
      # 0-shot
      lm_eval --model vllm \
         --model_args "pretrained=meta-llama/Llama-3.2-1B-Instruct,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_length=8192" \
         --tasks GRFNS2023 \
         --num_fewshot 0 \
         --batch_size auto \
         --output_path results \
         --hf_hub_log_args "hub_results_org=TheFinAI,results_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
         --log_samples \
         --apply_chat_template \
         --include_path ./tasks
   
      # 5-shot
      lm_eval --model vllm \
          --model_args "pretrained=Qwen/Qwen2.5-72B-Instruct,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_length=8192" \
          --tasks GRFNS2023 \
            --num_fewshot 5 \
          --batch_size auto \
          --output_path results \
          --hf_hub_log_args "hub_results_org=TheFinAI,results_repo_name=lm-eval-results-gr-5shot,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
          --log_samples \
          --apply_chat_template \
          --include_path ./tasks
      ```

## Results
   Evaluation results will be saved in the following locations:
      - **Local directory:** `FinBen/results/`
      - **Hugging Face Hub:** Defined in `results_repo_name` under `hub_results_org`.

