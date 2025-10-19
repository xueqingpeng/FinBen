import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from lib.tools import Tools
from bert_score import score as bert_score

def evaluate_predictions_to_dataframe(pred_dir, ground_truths, model_name="gpt-4o",language = 'en'):
    tools = Tools()
    records = []

    for i in ground_truths.index:
        gt = ground_truths.loc[i]
        if pd.isna(gt) or not isinstance(gt, str):
            print(f"⚠️ Skipping index {i}: ground truth is None or not a string")
            continue

        pred_path = os.path.join(pred_dir, f"{model_name}_pred_{i}.txt")
        if not os.path.exists(pred_path):
            print(f"⚠️ Missing: {pred_path}")
            continue

        with open(pred_path, "r", encoding="utf-8") as f:
            pred = f.read().strip()

        jaccard, masi, lev, cer, bleu, wer = tools.compute_distances(pred, gt)

        try:
            if language == 'en':
                P, R, F1 = bert_score([pred], [gt], lang="en", rescale_with_baseline=False)
            elif language == 'es':
                P, R, F1 = bert_score([pred], [gt], lang="es", rescale_with_baseline=False)
            else:
                print('Not a valid language, please try again')
                return language
            bert_f1 = F1.item()
        except Exception as e:
            print(f"⚠️ BERTScore error on index {i}: {e}")
            bert_f1 = None

        records.append({
            "index": i,
            "ground_truth": gt,
            "prediction": pred,
            "BLEU": bleu,
            "CER": cer,
            "WER": wer,
            "Levenshtein": lev,
            "Jaccard": jaccard,
            "MASI": masi,
            "BERTScore_F1": bert_f1,
            "Model": model_name
        })

    return pd.DataFrame(records)


def plot_violin(df_eval, output_prefix="llm_eval",language = 'en'):
    if df_eval.empty:
        print("❌ Evaluation DataFrame is empty — cannot plot.")
        return
    if language == 'en':
        os.makedirs("hyr_results/eval_plots", exist_ok=True)
        for metric in ["BLEU", "BERTScore_F1"]:
            plt.figure(figsize=(10, 5))
            sns.violinplot(x="Model", y=metric, data=df_eval, palette="Set2", cut=0)
            plt.title(f"{metric} Score Distribution")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"hyr_results/eval_plots/{output_prefix}_{metric.lower()}_violin.png", dpi=300)
            plt.close()
            print(f"✅ Saved: {metric} violin plot → hyr_results/eval_plots/{output_prefix}_{metric.lower()}_violin.png")
    elif language == 'es':
        os.makedirs("hyr_results/eval_plots_spanish", exist_ok=True)
        for metric in ["BLEU", "BERTScore_F1"]:
            plt.figure(figsize=(10, 5))
            sns.violinplot(x="Model", y=metric, data=df_eval, palette="Set2", cut=0)
            plt.title(f"{metric} Score Distribution")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"hyr_results/eval_plots_spanish/{output_prefix}_{metric.lower()}_violin.png", dpi=300)
            plt.close()
            print(f"✅ Saved: {metric} violin plot → hyr_results/eval_plots_spanish/{output_prefix}_{metric.lower()}_violin.png")
    else:
            print('Not a valid language, please try again')
            return language

    


def run_eval_and_plot(parquet_path, pred_dir, model_name="gpt-4o", language = 'en' output_csv=None):
    df = pd.read_parquet(parquet_path)

    pred_indexes = []
    for fname in os.listdir(pred_dir):
        if fname.startswith(f"{model_name}_pred_") and fname.endswith(".txt"):
            try:
                idx = int(fname.replace(f"{model_name}_pred_", "").replace(".txt", ""))
                pred_indexes.append(idx)
            except:
                continue

    # Keep only matching rows from the DataFrame
    df = df.loc[df.index.intersection(pred_indexes)]

    df_eval = evaluate_predictions_to_dataframe(pred_dir, df["matched_html"], model_name=model_name,language = language)
    if output_csv:
        df_eval.to_csv(output_csv, index=False)
        print(f"✅ Evaluation saved to CSV: {output_csv}")

    plot_violin(df_eval, output_prefix=model_name, language = language)
    print(df_eval.head())
    return df_eval


def main():
    run_eval_and_plot(
        parquet_path="hyr_ocr_process/spanish_output_parquet/spanish_batch_0000.parquet",
        pred_dir="hyr_results/predictions_spanish/gpt-4o_zero-shot_financial",
        model_name="gpt-4o",
        output_csv="hyr_results/eval_spanish_gpt_4o.csv"
    )

if __name__ == '__main__':
    main()
