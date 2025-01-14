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

Navigate to the FinBen directory:
```bash
cd FinBen/
```

### GPT Model Evaluation
```bash
lm_eval --model openai-chat-completions --model_args "model=gpt-4o" --tasks GRQAGen --output_path results --use_cache ./cache --log_samples --apply_chat_template --include_path ./tasks
```

### Small Model Evaluation
```bash
lm_eval --model vllm --model_args "pretrained=meta-llama/Llama-3.2-1B-Instruct" --tasks GRMultifin --num_fewshot 5 --device cuda:0 --batch_size 8 --output_path results --log_samples --apply_chat_template --include_path ./tasks
```
The accuracy achieved for this command is as follows: 0.3704 for 0-shot and 0.3889 for 5-shot.

### Large Model Evaluation
1. Set the VLLM worker multiprocessing method:
   ```bash
   export VLLM_WORKER_MULTIPROC_METHOD="spawn"
   ```
2. Run large model evaluation:
   ```bash
   lm_eval --model vllm --model_args "pretrained=google/gemma-2-27b-it,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_model_len=1024" --tasks GRQA --batch_size auto --output_path results --log_samples --apply_chat_template --include_path ./tasks
   ```
   ```bash
   lm_eval --model vllm --model_args "pretrained=meta-llama/Llama-3.2-1B-Instruct,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_length=8192" --tasks GRFNS2023 --batch_size auto --output_path results --log_samples --apply_chat_template --include_path ./tasks
   ```

Results will be saved to `FinBen/results/`.

