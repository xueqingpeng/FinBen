import evaluate
import re
from typing import List, Dict, Any


# ==========================
# Rouge-1
# ==========================

def rouge1(items):
    """
    # passthrough for efficiency
    """
    return items


def rouge1_agg(items):
    """
    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    rouge_scorer = evaluate.load("rouge")
    return rouge_scorer.compute(predictions=preds, references=refs)["rouge1"]


# ==========================
# BERTScore (F1)
# ==========================

def berts(items):
    """
    Passthrough for BERTScore.
    Actual scoring is done in berts_f1_agg.
    """
    return items


def berts_f1_agg(items):
    """
    Aggregation function for BERTScore F1.
    Higher is better.
    items: list of (reference, prediction)
    """
    if not items:
        return 0.0

    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]

    bertscore = evaluate.load("bertscore")
    scores = bertscore.compute(
        predictions=preds,
        references=refs,
        model_type="roberta-large",  # English model
        lang="en",
    )
    f1_list = scores["f1"]
    return float(sum(f1_list) / len(f1_list))


# ==========================
# Numeric Consistency Utilities
# ==========================

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


def _numeric_consistency_for_sample(
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


def numeric_consistency_agg(items: List[float]) -> float:
    """
    Aggregation function for numeric consistency.
    items: list of per-sample numeric consistency scores.
    Returns the average score.
    """
    if not items:
        return 0.0
    return float(sum(items) / len(items))


# ==========================
# PolyFiQA process_results
# Returns Rouge-1 / BERTScore / Numeric Consistency
# ==========================

def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """
    PolyFiQA per-sample evaluation function.

    doc: JSON dict with at least:
        - doc["query"]: full context (instruction + tables + multilingual news + question)
        - doc["answer"]: ground-truth answer

    results: list containing the model's generated answer.
    """
    context = doc["query"]
    gold_answer = doc["answer"]
    pred_answer = results[0]

    # 1. Numeric consistency score for this sample
    nc_score = _numeric_consistency_for_sample(context, pred_answer)

    # 2. Rouge-1 and BERTScore pairs (reference, prediction)
    rouge_pair = (gold_answer, pred_answer)
    berts_pair = (gold_answer, pred_answer)

    return {
        "numeric_consistency": float(nc_score),  # aggregation: numeric_consistency_agg
        "rouge1": rouge_pair,                    # aggregation: rouge1_agg
        "bertscore": berts_pair,                 # aggregation: berts_f1_agg
    }