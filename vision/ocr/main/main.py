from lib.agent import Agent
from lib.tools import Tools
import pandas as pd
from tqdm import tqdm
import os
import time
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()


def evaluate(
    model_name="gpt-4o", 
    experiment_tag="zero-shot", 
    language="en", 
    local_version=True, 
    local_dir="./OCR_Task", 
    sample=None
):
    LOCAL_FILES = {
        "en": ["OCR_DATA/local_file_version/EnglishOCR_v2.parquet"],
        "es": [
            "OCR_DATA/local_file_version/spanish_batch_0000.parquet",
            "OCR_DATA/local_file_version/spanish_batch_0001.parquet",
            "OCR_DATA/local_file_version/spanish_batch_0002.parquet",
        ],
        "gr": ["OCR_DATA/local_file_version/GreekOCR_v1.parquet"],
        "jp": ["OCR_DATA/local_file_version/JapaneseOCR_v1.parquet"],
    }
    REMOTE_FILES = {
        "en": ["OCR_DATA/base64_encoded_version/EnglishOCR_3000_000.parquet"],
        "es": ["OCR_DATA/base64_encoded_version/SpanishOCR_3000_000.parquet"],
        "gr": ["OCR_DATA/base64_encoded_version/GreekOCR_full_000.parquet"],
        "jp": ["OCR_DATA/base64_encoded_version/JapaneseOCR_full_000.parquet"],
    }

    valid_langs = {"en", "es", "gr", "jp"}
    if language not in valid_langs:
        raise ValueError(f"Invalid language '{language}'. Choose from {sorted(valid_langs)}.")
        
    if local_version:
        paths = [os.path.join(local_dir, p) for p in LOCAL_FILES[language]]
        dfs = [pd.read_parquet(p) for p in paths]
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    else:
        ds = load_dataset("TheFinAI/OCR_Task", data_files=REMOTE_FILES[language])
        df = ds["train"].to_pandas()
    
    experiment_folder = f'./results/{language}/{model_name.replace("/", "-")}_{experiment_tag}'
    os.makedirs(experiment_folder, exist_ok=True)

    # Get predicted indices from filenames
    predicted_indices = set()
    if os.path.exists(experiment_folder):
        for fname in os.listdir(experiment_folder):
            if fname.startswith(f"pred_") and fname.endswith(".txt"):
                try:
                    idx = int(fname.replace(f"pred_", "").replace(".txt", ""))
                    predicted_indices.add(idx)
                except:
                    continue

    # Filter out completed predictions
    # df = df.iloc[900:1000]
    df = df[~df.index.isin(predicted_indices)]

    # Apply sample AFTER filtering
    if sample:
        df = df.head(sample)  # get sample

    agent = Agent(model_name)

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {model_name}"):
        image_path = row.get("image_path", row.get("image"))
        image_path = os.path.join(local_dir, image_path).replace("./", "").replace("Japanese/", "")
        output_file = os.path.join(experiment_folder, f"pred_{i}.txt")

        try:
            result = agent.draft(image_path, local_version=local_version)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)
            # time.sleep(1)
        except Exception as e:
            print(f"⚠️ Error on index {i}: {e}")
            continue

def main():
    models = [
        "gpt-4o"
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        # "google/gemma-3-4b-it",
        "google/gemma-3-27b-it",
        # "Qwen/Qwen2.5-Omni-7B",
        # "TheFinAI/FinLLaVA",
        # "Qwen/Qwen-VL-Max",
        "liuhaotian/llava-v1.6-vicuna-13b",
        "deepseek-ai/deepseek-vl-7b-chat",
    ]
    languages = [
        # "en", 
        # "es",
        "gr",
        "jp"
    ]

    for model in models:
        for language in languages:
            print(f"🟢 Start evaluating {model} in {language}")
            try:
                evaluate(
                    model_name=model,
                    experiment_tag="zero-shot",
                    language=language,
                    local_version = True, 
                    local_dir = "/gpfs/radev/project/xu_hua/xp83/OCR_Task", 
                    sample = 1000
                )
            except Exception as e:
                print(f"⚠️ Error with model {model} in {language}: {e}")
                continue

if __name__ == '__main__':
    main()
