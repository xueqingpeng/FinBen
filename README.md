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

Set your Hugging Face token as an environment variable:
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

3. Run evaluation:
   ### Important Notes on Evaluation
   - **0-shot setting:** Use `num_fewshot=0` and `lm-eval-results-gr-0shot` as the results repository.
   - **5-shot setting:** Use `num_fewshot=5` and `lm-eval-results-gr-5shot` as the results repository.
   - **Base models:** Remove `apply_chat_template`.
   - **Instruction models:** Use `apply_chat_template`.

   ### For gr Tasks
   Execute the following command:
   ```bash
   lm_eval --model vllm \
      --model_args "pretrained=meta-llama/Llama-3.2-1B-Instruct,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_model_len=1024" \
      --tasks gr \
      --num_fewshot 5 \
      --batch_size auto \
      --output_path results \
      --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-gr-5shot,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
      --log_samples \
      --apply_chat_template \
      --include_path ./tasks
   ```

   ### For gr_long Task
   Execute the following command:
   ```bash
   lm_eval --model vllm \
       --model_args "pretrained=Qwen/Qwen2.5-72B-Instruct,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_length=8192" \
       --tasks gr_long \
       --num_fewshot 5 \
       --batch_size auto \
       --output_path results \
       --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-gr-5shot,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
       --log_samples \
       --apply_chat_template \
       --include_path ./tasks
   ```

## Results
Evaluation results will be saved in the following locations:
- **Local Directory:** `FinBen/results/`
- **Hugging Face Hub:** As defined in `details_repo_name` under `hub_results_org`.

---

## Notes

The `lm-eval-results` is directly linked to our Greek leaderboard. If you have added a new model to this repo that is not included in `FinBen/aggregate.py`, please provide me with all the necessary information.

   ### Relevant Resources
   You can find related information at the following links:
   - [Open LLM Leaderboard on Hugging Face](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=qwen2.5-1.5b-instruct)
   - [Open AI Models](https://mot.isitopen.ai/)

   ### Model Information Template
   Please include the following details in your submission for the new model:

   ```python
   "ilsp/Meltemi-7B-Instruct-v1.5": {
      # "Architecture": "",
      "Hub License": "apache-2.0",
      "Hub ❤️": 17,
      "#Params (B)": 7.48,
      "Available on the hub": True,
      "MoE": False,
      # "generation": 0,
      "Base Model": "ilsp/Meltemi-7B-v1.5",
      "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
      "T": "💬",
      "full_model_name": '<a target="_blank" href="https://huggingface.co/ilsp/Meltemi-7B-Instruct-v1.5" style="color: var(--link-text-color); text-decoration: underline; text-decoration-style: dotted;">meta-llama/Llama-3.2-1B-Instruct</a>',
      # "co2_kg_per_s": 0
   }
   ```

   For any parameters that you cannot find, it's perfectly fine to comment them out.

   Thank you for your contributions!
