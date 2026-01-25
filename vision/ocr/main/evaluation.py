import os
import pandas as pd
from tqdm import tqdm
from evaluate import load
from lib.tools import Tools
import re
import html
from bs4 import BeautifulSoup
import Levenshtein
import evaluate
from typing import List, Dict, Any

rouge = load("rouge")


def html_to_text(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    # 不是HTML也直接做基本清洗
    if "<" not in s or ">" not in s:
        t = html.unescape(s)
        t = re.sub(r"\s+", " ", t).strip()
        return t
    
    # 解析 HTML
    soup = BeautifulSoup(s, "lxml")  # 若无lxml也可用 "html.parser"
    # 去掉脚本/样式等
    for tag in soup(["script", "style", "noscript", "template", "iframe"]):
        tag.decompose()
    # 提取纯文本
    t = soup.get_text(separator=" ", strip=True)
    # 解码实体 & 规范空白
    t = html.unescape(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def calculate_edit_distance(reference, hypothesis):
    if not reference and not hypothesis:
        return 0.0

    if not reference or not hypothesis:
        return 1.0

    distance = Levenshtein.distance(reference, hypothesis)
    max_len = max(len(reference), len(hypothesis))

    return distance / max_len if max_len > 0 else 0.0

def calculate_bertscore(reference, prediction):
    bertscore = evaluate.load("bertscore")
    scores = bertscore.compute(
        predictions=[prediction],
        references=[reference],
        model_type="roberta-large",  # English model
        lang="en",
    )
    f1_list = scores["f1"]
    return float(sum(f1_list) / len(f1_list))

_NUM_REGEX = re.compile(
    r"(?<![\w.])(?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)(?P<pct>%?)"
)

def _extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from a text string.
    Supports:
      - Thousands separators: 1,234.56
      - Percentages: 15%
    Percentages are normalized to decimal form (e.g., 15% → 0.15).
    """
    nums: List[float] = []
    for m in _NUM_REGEX.finditer(text):
        raw = m.group("num")
        is_pct = m.group("pct") == "%"

        val = float(raw.replace(",", ""))
        if is_pct:
            val /= 100.0

        nums.append(val)

    return nums

def calculate_numeric_consistency(
    context: str,
    answer: str,
    rel_tol: float = 0.02,
    abs_tol: float = 1e-6,
) -> float:
    """
    Compute numeric consistency for a single sample.

    Steps:
      1. Extract numeric set C from context.
      2. Extract numeric set A from answer.
      3. For each number in A, find the closest number in C.
      4. A match succeeds if relative error <= rel_tol.
      5. Score = (# matched numbers) / (# numbers in answer)

    Corner cases:
      - If answer has no numbers → score = 1.0
      - If context has no numbers but answer does → score = 0.0
    """
    ctx_nums = _extract_numbers(context)
    ans_nums = _extract_numbers(answer)

    if not ans_nums:
        return 1.0
    if not ctx_nums:
        return 0.0

    matched = 0
    for a in ans_nums:
        best_rel_err = None
        for c in ctx_nums:
            diff = abs(a - c)
            denom = max(abs(a), abs(c), abs_tol)
            rel_err = diff / denom

            if best_rel_err is None or rel_err < best_rel_err:
                best_rel_err = rel_err

        if best_rel_err is not None and (
            best_rel_err <= rel_tol or abs(a) <= abs_tol
        ):
            matched += 1

    return matched / len(ans_nums)

def evaluate_rouge(pred_dir, ground_truths, model_name="gpt-4o",lang='en'):
    records = []

    for i in tqdm(ground_truths.index, total=len(ground_truths), desc="Evaluating ROUGE"):
        gt = ground_truths.loc[i]
        if pd.isna(gt) or not isinstance(gt, str):
            continue

        pred_path = os.path.join(pred_dir, f"pred_{i}.txt")
        if not os.path.exists(pred_path):
            continue

        with open(pred_path, "r", encoding="utf-8") as f:
            pred = f.read().strip()
            clean_pred = pred
            if lang == "en":
                clean_pred = html_to_text(pred)
                gt = html_to_text(gt)
            # if lang != "es":
            #     import re
            #     clean_pred = re.sub(r"<[^>]+>", " ", pred)
            #     clean_pred = re.sub(r"\s+", " ", clean_pred).strip()

        try:
            rouge_score = rouge.compute(predictions=[clean_pred], references=[gt], use_stemmer=True)
            rouge_1_f1 = float(rouge_score["rouge1"])
            rouge_l_f1 = float(rouge_score["rougeL"])
            edit_distance = calculate_edit_distance(gt, clean_pred)
            bertscore_f1 = calculate_bertscore(gt, clean_pred)
            numeric_consistency = calculate_numeric_consistency(gt, clean_pred)
        except Exception as e:
            print(f"ROUGE error on index {i}: {e}")
            rouge_1_f1 = None
            rouge_l_f1 = None
            edit_distance = None
            bertscore_f1 = None
            numeric_consistency = None

        records.append({
            "index": i,
            "ground_truth": gt,
            "prediction": pred,
            "ROUGE-1": rouge_1_f1,
            "ROUGE-L": rouge_l_f1,
            "Edit Distance": edit_distance,
            "BERTScore-F1": bertscore_f1,
            "Numeric Consistency": numeric_consistency,
            "Model": model_name,
            "Language": lang
        })

    df_eval = pd.DataFrame(records)

    # Create table format output
    result = {
        'language': [lang],
        'model': [model_name], 
        'sample_size': [len(df_eval)],
        'rouge1': [df_eval['ROUGE-1'].mean()],
        'rougeL': [df_eval['ROUGE-L'].mean()],
        'edit_distance': [df_eval['Edit Distance'].mean()],
        'bertscore_f1': [df_eval['BERTScore-F1'].mean()],
        'numeric_consistency': [df_eval['Numeric Consistency'].mean()]
    }
    df_result = pd.DataFrame(result)
    print(df_result.to_string(index=False, float_format='%.4f'))

    return df_eval, df_result

def run_rouge_eval(
    model_name="gpt-4o",
    experiment_tag="zero-shot",
    language="en",
    local_version = True, 
    local_dir = "/gpfs/radev/project/xu_hua/xp83/OCR_Task", 
):
    LOCAL_FILES = {
        "smallocr": ["OCR_DATA/FinOCRBench_Task1_input.csv"],
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

    valid_langs = {"smallocr", "en", "es", "gr", "jp"}
    if language not in valid_langs:
        raise ValueError(f"Invalid language '{language}'. Choose from {sorted(valid_langs)}.")
        
    if local_version:
        paths = [os.path.join(local_dir, p) for p in LOCAL_FILES[language]]
        if language == "smallocr":
            df = pd.read_csv(paths[0])
        else:
            dfs = [pd.read_parquet(p) for p in paths]
            df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    else:
        if language == "smallocr":
            ds = load_dataset(
                "csv",
                data_files="https://huggingface.co/datasets/TheFinAI/FinCriticalED/resolve/main/raw_input_additional.csv"
            )
            df = ds["train"].to_pandas()
        else:
            ds = load_dataset("TheFinAI/OCR_Task", data_files=REMOTE_FILES[language])
            df = ds["train"].to_pandas()

    pred_dir = f'./results/{language}/{model_name.replace("/", "-")}_{experiment_tag}'

    # Extract indices from prediction files
    pred_indexes = []
    for fname in os.listdir(pred_dir)[:60]:
        if fname.startswith(f"pred_") and fname.endswith(".txt"):
            try:
                idx = int(fname.replace(f"pred_", "").replace(".txt", ""))
                pred_indexes.append(idx)
            except:
                continue

    df = df.loc[df.index.intersection(pred_indexes)]
    # df_eval, df_result = evaluate_rouge(pred_dir, df["matched_html"], model_name=model_name, lang=language)
    df_eval, df_result = evaluate_rouge(pred_dir, df["data.matched_html"], model_name=model_name, lang=language)

    eval_fp = f'./results/{language}/{model_name.replace("/", "-")}_{experiment_tag}_eval.csv'
    result_fp = f'./results/{language}/{model_name.replace("/", "-")}_{experiment_tag}_result.csv'
    df_eval.to_csv(eval_fp, index=False)
    df_result.to_csv(result_fp, index=False)
    print(f"✅ Evaluation saved to CSV")
    return df_eval

def main():
    models = [
        "gpt-4o",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        # "google/gemma-3-4b-it",
        "google/gemma-3-27b-it",
        # "Qwen/Qwen2.5-Omni-7B",
        "deepseek-ai/deepseek-vl-7b-chat",
        "liuhaotian/llava-v1.6-vicuna-13b",
        # # "TheFinAI/FinLLaVA",
        # # "Qwen/Qwen-VL-Max",
        
        # "Qwen/Qwen2.5-VL-72B-Instruct",
        # "google/gemma-3n-E4B-it",
        # "gpt-5",
    ]
    languages = [
        "smallocr",
        # "en", 
        # "es",
        # "gr",
        # "jp"
    ]

    for model in models:
        for language in languages:
            print(f"🟢 Start evaluating {model} in {language}")
            try:
                run_rouge_eval(
                    model_name=model,
                    experiment_tag="zero-shot",
                    language=language,
                    local_version = True, 
                    local_dir = "/gpfs/radev/project/xu_hua/xp83/OCR_Task", 
                )
            except Exception as e:
                print(f"⚠️ Error with model {model} in {language}: {e}")
                continue

if __name__ == '__main__':
    main()
