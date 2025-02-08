import json
import os
from datetime import datetime

import numpy as np
from datasets import Dataset
from huggingface_hub import snapshot_download


MODEL_DICT = {
    "gpt-4": {
        "Architecture": "Transformer",
        "Hub License": "openai",
        "Hub ❤️": 0,
        "#Params (B)": 1800,
        "Available on the hub": False,
        "MoE": True,
        "generation": 0,
        "Base Model": "GPT-4",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://openai.com/index/gpt-4/" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">GPT-4</a>',
        "co2_kg_per_s": 0.0023,
    },
    "gpt-4o": {
        "Architecture": "Transformer",
        "Hub License": "openai",
        "Hub ❤️": 0,
        "#Params (B)": 2500.0,
        "Available on the hub": False,
        "MoE": True,
        "generation": 0,
        "Base Model": "GPT-4o",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://openai.com/index/gpt-4o/" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">GPT-4</a>',
        "co2_kg_per_s": 0.0023,
    },
    "gpt-4o-mini": {
        "Architecture": "Transformer",
        "Hub License": "openai",
        "Hub ❤️": 0,
        "#Params (B)": 500.0,
        "Available on the hub": False,
        "MoE": True,
        "generation": 0,
        "Base Model": "GPT-4o-mini",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://openai.com/index/gpt-4o-mini/" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">GPT-4</a>',
        "co2_kg_per_s": 0.0023,
    },
    "gpt-3.5-turbo-0125": {
        "Architecture": "Transformer",
        "Hub License": "openai",
        "Hub ❤️": 0,
        "#Params (B)": 400.0,
        "Available on the hub": False,
        "MoE": True,
        "generation": 0,
        "Base Model": "GPT-3.5-turbo",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://openai.com/index/gpt-3.5/" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">GPT-4</a>',
        "co2_kg_per_s": 0.0023,
    },
    "deepseek-chat": {
        # "Architecture": "Innovative Load Balancing Strategy and Training Objective",
        "Hub License": "deepseek",
        "Hub ❤️": 1630,
        "#Params (B)": 685,
        "Available on the hub": False,
        "MoE": True,
        # "generation": 0,
        # "Base Model": "",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://huggingface.co/deepseek-ai/DeepSeek-V3" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">GPT-4</a>',
        # "co2_kg_per_s": 0,
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "Architecture": "LlamaForCausalLM",
        "Hub License": "llama3.2",
        "Hub ❤️": 684,
        "#Params (B)": 1.24,
        "Available on the hub": True,
        "MoE": False,
        "generation": 0,
        "Base Model": "meta-llama/Llama-3.2-1B-Instruct",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">meta-llama/Llama-3.2-1B-Instruct</a>',
        "co2_kg_per_s": 0.40,
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "Architecture": "LlamaForCausalLM",
        "Hub License": "llama3",
        "Hub ❤️": 3749,
        "#Params (B)":8.03,
        "Available on the hub": True,
        "MoE": False,
        "generation": 0,
        "Base Model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">meta-llama/Llama-3.2-1B-Instruct</a>',
        "co2_kg_per_s": 0.80,
    },
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "Architecture": "LlamaForCausalLM",
        "Hub License": "llama3",
        "Hub ❤️": 1451,
        "#Params (B)":70.554,
        "Available on the hub": True,
        "MoE": False,
        "generation": 1,
        "Base Model": "meta-llama/Meta-Llama-3-70B",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">meta-llama/Llama-3.2-1B-Instruct</a>',
        "co2_kg_per_s": 18.24,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "Architecture": "Qwen2ForCausalLM",
        "Hub License": "apache-2.0",
        "Hub ❤️": 270,
        "#Params (B)":1.5,
        "Available on the hub": True,
        "MoE": False,
        "generation": 1,
        "Base Model": "Qwen/Qwen2.5-1.5B",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://huggingface.co/Qwen2.5-1.5B-Instruct" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">meta-llama/Llama-3.2-1B-Instruct</a>',
        "co2_kg_per_s": 0.69,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "Architecture": "Qwen2ForCausalLM",
        "Hub License": "apache-2.0",
        "Hub ❤️": 413,
        "#Params (B)":7.616,
        "Available on the hub": True,
        "MoE": False,
        "generation": 1,
        "Base Model": "Qwen/Qwen2.5-7B",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">meta-llama/Llama-3.2-1B-Instruct</a>',
        "co2_kg_per_s": 2.17,
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "Architecture": "Qwen2ForCausalLM",
        "Hub License": "apache-2.0",
        "Hub ❤️": 170,
        "#Params (B)":32.764,
        "Available on the hub": True,
        "MoE": False,
        "generation": 1,
        "Base Model": "Qwen/Qwen2.5-32B",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">meta-llama/Llama-3.2-1B-Instruct</a>',
        "co2_kg_per_s": 5.75,
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "Architecture": "Qwen2ForCausalLM",
        "Hub License": "other",
        "Hub ❤️": 663,
        "#Params (B)":72.706,
        "Available on the hub": True,
        "MoE": False,
        "generation": 1,
        "Base Model": "Qwen/Qwen2.5-72B",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">meta-llama/Llama-3.2-1B-Instruct</a>',
        "co2_kg_per_s": 33.01,
    },
    "google/gemma-2-2b-it": {
        "Architecture": "InternLM2ForCausalLM",
        "Hub License": "gemma",
        "Hub ❤️": 854,
        "#Params (B)":2.614,
        "Available on the hub": True,
        "MoE": False,
        "generation": 1,
        "Base Model": "google/gemma-2-2b",
        "Type": "🔶 fine-tunedondomain-specificdatasets",
        "T": "🔶",
        "full_model_name": '<a target="_blank" href="https://huggingface.co/google/gemma-2-2b-it" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">meta-llama/Llama-3.2-1B-Instruct</a>',
        "co2_kg_per_s": 1.23,
    },
    "google/gemma-2-9b-it": {
        "Architecture": "Gemma2ForCausalLM",
        "Hub License": "gemma",
        "Hub ❤️": 613,
        "#Params (B)":9,
        "Available on the hub": True,
        "MoE": False,
        "generation": 1,
        "Base Model": "google/gemma-2-9b",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://huggingface.co/google/gemma-2-9b-it" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">meta-llama/Llama-3.2-1B-Instruct</a>',
        "co2_kg_per_s": 5.01,
    },
    "google/gemma-2-27b-it": {
        "Architecture": "Gemma2ForCausalLM",
        "Hub License": "gemma",
        "Hub ❤️": 491,
        "#Params (B)":27.227,
        "Available on the hub": True,
        "MoE": False,
        "generation": 1,
        "Base Model": "google/gemma-2-27b",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://huggingface.co/google/gemma-2-27b-it" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">meta-llama/Llama-3.2-1B-Instruct</a>',
        "co2_kg_per_s": 4.83,
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "Architecture": "MistralForCausalLM",
        "Hub License": "apache-2.0",
        "Hub ❤️": 1233,
        "#Params (B)":7.248,
        "Available on the hub": True,
        "MoE": False,
        "generation": 1,
        "Base Model": "mistralai/Mistral-7B-v0.3",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">meta-llama/Llama-3.2-1B-Instruct</a>',
        "co2_kg_per_s": 0.54,
    },
    "TheFinAI/FinLLaMA-instruct": {
        # "Architecture": "",
        # "Hub License": "",
        "Hub ❤️": 9,
        "#Params (B)":7.5,
        "Available on the hub": True,
        # "MoE": False,
        # "generation": 0,
        "Base Model": "TheFinAI/FinLLaMA",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://huggingface.co/TheFinAI/FinLLaMA-instruct" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">meta-llama/Llama-3.2-1B-Instruct</a>',
        # "co2_kg_per_s": 0,
    },
    "ilsp/Meltemi-7B-Instruct-v1.5": {
        # "Architecture": "",
        "Hub License": "apache-2.0",
        "Hub ❤️": 17,
        "#Params (B)":7.48,
        "Available on the hub": True,
        "MoE": False,
        # "generation": 0,
        "Base Model": "ilsp/Meltemi-7B-v1.5",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://huggingface.co/ilsp/Meltemi-7B-Instruct-v1.5" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">meta-llama/Llama-3.2-1B-Instruct</a>',
        # "co2_kg_per_s": 0,
    },
}

METRIC_DICT = {
    "GRMultifinGen": {
        "task_name": "MultiFin",
        "random_baseline": 1.0 / 6.0,
        "key": "exact_match,score-first",
    },
    "GRMultifin": {
        "task_name": "MultiFin",
        "random_baseline": 1.0 / 6.0,
        "key": "acc_norm,none",
    },
    "GRQAGen": {
        "task_name": "QA",
        "random_baseline": 1.0 / 3.8,
        "key": "exact_match,score-first",
    },
    "GRQA": {
        "task_name": "QA",
        "random_baseline": 1.0 / 3.8,
        "key": "acc_norm,none",
    },
    "GRFNS2023": {
        "task_name": "FNS",
        "random_baseline": 0.0,
        "key": "rouge1,none",
    },
    "GRFINNUM": {
        "task_name": "GRFINNUM",
        "random_baseline": 0.0,
        "key": "f1",
    },
    "GRFINTEXT": {
        "task_name": "GRFINTEXT",
        "random_baseline": 0.0,
        "key": "f1",
    },
}


def normalize_within_range(value, lower_bound=0, higher_bound=1):
    return (np.clip(value - lower_bound, 0, None)) / (higher_bound - lower_bound) * 100


def calculate_co2_emissions(
    total_evaluation_time_seconds: float | None, model_name: str
) -> float:
    model_info = MODEL_DICT.get(model_name, {})
    if "co2_kg_per_s" in model_info:
        return (
            total_evaluation_time_seconds * model_info["co2_kg_per_s"]
            if total_evaluation_time_seconds
            else -1
        )

    if total_evaluation_time_seconds is None or total_evaluation_time_seconds <= 0:
        return -1

    # Power consumption for 8 A100 GPUs in kilowatts (kW)
    power_consumption_kW = 0.3

    # Carbon intensity in grams CO₂ per kWh in Florida
    carbon_intensity_g_per_kWh = 360

    # Convert evaluation time to hours
    total_evaluation_time_hours = total_evaluation_time_seconds / 3600

    # Calculate energy consumption in kWh
    energy_consumption_kWh = power_consumption_kW * total_evaluation_time_hours

    # Calculate CO₂ emissions in grams
    co2_emissions_g = energy_consumption_kWh * carbon_intensity_g_per_kWh

    # Convert grams to kilograms
    return co2_emissions_g / 1000


def load_dataset_from_huggingface(repo_id):
    dataset_path = snapshot_download(repo_id, repo_type="dataset")
    dataset = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.startswith("result") and file.endswith(".json"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    dataset.append(json.load(f))
    return dataset


def aggregate_results(dataset):
    aggregated = {}

    sorted_dataset = sorted(
        dataset, key=lambda x: x.get("date", 0), reverse=True
    )  # Sort by date in descending order

    for data in sorted_dataset:
        original_task_name = list(data["results"].keys())[0]
        task_mapping = METRIC_DICT.get(
            original_task_name, {"task_name": original_task_name, "random_baseline": 0}
        )
        task_name = task_mapping["task_name"]
        lower_bound = task_mapping["random_baseline"]
        task_key = task_mapping["key"]

        task_data = data["results"].get(original_task_name, {})
        config_data = data["configs"].get(original_task_name, {})
        model_name = data.get("model_name", original_task_name)
        print(model_name, task_data)
        model_info = MODEL_DICT.get(model_name, {})

        if model_name not in aggregated:
            aggregated[model_name] = {
                "eval_name": model_name + model_info.get("Precision", "bfloat16"),
                "Precision": model_info.get("Precision", "bfloat16"),
                "Type": model_info.get(
                    "Type", data.get("config", {}).get("model", "Unknown")
                ),
                "T": model_info.get("T", "Unknown"),
                "Weight type": "Original",
                "Architecture": model_info.get(
                    "Architecture", config_data.get("output_type", "Unknown")
                ),
                "Model": model_info.get("full_model_name", model_name),
                "fullname": model_name,
                "Model sha": data.get("task_hashes", {}).get(original_task_name, ""),
                "Hub License": model_info.get(
                    "Hub License",
                ),
                "Hub ❤️": model_info.get(
                    "Hub ❤️",
                ),
                "#Params (B)": model_info.get(
                    "#Params (B)",
                ),
                "Available on the hub": model_info.get("Available on the hub", False),
                "MoE": model_info.get("MoE", False),
                "Flagged": False,
                "Chat Template": True,
                "CO₂ cost (kg)": calculate_co2_emissions(
                    float(
                        data.get("total_evaluation_time_seconds", 0.6095870820618575)
                    ),
                    model_name,
                ),
                task_name + " Raw": None,
                task_name: None,
                "Merged": False,
                "Official Providers": False,
                "Upload To Hub Date": datetime.utcfromtimestamp(
                    data.get("date", 1736341347.1137679)
                ).strftime("%Y-%m-%d"),
                "Submission Date": datetime.utcfromtimestamp(data.get("date")).strftime(
                    "%Y-%m-%d"
                ),
                "Generation": model_info.get("generation", 0),
                "Base Model": model_info.get("Base Model", model_name),
            }

        raw_score = task_data.get(task_key, 0)
        normalized_score = normalize_within_range(
            raw_score, lower_bound=lower_bound, higher_bound=1
        )

        aggregated[model_name][task_name + " Raw"] = raw_score
        aggregated[model_name][task_name] = normalized_score

    for model_name, model_data in aggregated.items():
        task_scores = [
            model_data[key.replace(" Raw", "")]
            for key in model_data.keys()
            if key.endswith(" Raw")
            and model_data.get(key.replace(" Raw", ""), None) is not None
        ]
        aggregated[model_name]["Average ⬆️"] = (
            sum(task_scores) / len(task_scores) if task_scores else None
        )

    return aggregated


def update_greek_contents(aggregated_results, repo_id="TheFinAI/greek-contents"):
    dataset = Dataset.from_list([value for value in aggregated_results.values()])
    dataset.push_to_hub(repo_id)


# Example Usage
repo_id = "TheFinAI/lm-eval-results-private"  # Update with actual repo ID
json_files_dataset = load_dataset_from_huggingface(repo_id)
aggregated_results = aggregate_results(json_files_dataset)
print(json.dumps(aggregated_results, indent=2))
update_greek_contents(aggregated_results)
