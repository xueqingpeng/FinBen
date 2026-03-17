from sklearn.metrics import f1_score

def macro_f1_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="macro")
    return fscore



import re
import json
import ast
from typing import List, Dict, Any, Tuple, Union


# ==========================
# Utility functions
# ==========================

def _normalize_term_text(t: str) -> str:
    """
    Normalize a financial term:
      - trim leading/trailing whitespace
      - remove all whitespace characters (spaces, newlines, etc.)
    This tends to work well for Japanese financial terms.
    """
    if not isinstance(t, str):
        t = str(t)
    t = t.strip()
    t = re.sub(r"\s+", "", t)
    return t


def _extract_json_like_block(raw: str) -> Union[str, None]:
    """
    Extract the first JSON-like array block from model output.
    Matches:
      - starting with '[' and ending with ']'
      - across multiple lines
    """
    if not isinstance(raw, str):
        raw = str(raw)
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return None
    return m.group(0)


def _parse_term_structure(raw: Any) -> List[List[str]]:
    """
    Normalize gold/pred outputs into List[List[str]] format.

    Supports:
      - Python objects (list / list-of-lists)
      - JSON strings
      - Python literal strings (single quotes, etc.)
      - text containing extra content with an embedded "[ ... ]" block
    """
    # Already a list
    if isinstance(raw, list):
        obj = raw
    else:
        s = str(raw).strip()

        # Try JSON first
        try:
            obj = json.loads(s)
        except Exception:
            # Extract JSON-like substring
            block = _extract_json_like_block(s)
            if block is None:
                return []
            # Try JSON again
            try:
                obj = json.loads(block)
            except Exception:
                # Single quotes → double quotes
                block2 = block.replace("'", '"')
                try:
                    obj = json.loads(block2)
                except Exception:
                    # Last fallback: literal eval
                    try:
                        obj = ast.literal_eval(block)
                    except Exception:
                        return []

    # Normalize to List[List[str]]
    if isinstance(obj, list):
        if all(isinstance(x, str) for x in obj):
            # ["t1","t2"] -> [["t1","t2"]]
            obj = [obj]
        elif all(isinstance(x, list) for x in obj):
            pass
        else:
            # Mixed structures: only keep list[str] elements
            cleaned = []
            for x in obj:
                if isinstance(x, list) and all(isinstance(y, str) for y in x):
                    cleaned.append(x)
            obj = cleaned
    else:
        return []

    norm_struct: List[List[str]] = []
    for sub in obj:
        if not isinstance(sub, list):
            continue
        terms = []
        for t in sub:
            if not isinstance(t, str):
                t = str(t)
            terms.append(t)
        if terms:
            norm_struct.append(terms)

    return norm_struct


def _flatten_unique_terms(structs: List[List[List[str]]]) -> List[str]:
    """
    structs: [terms_for_sample_1, terms_for_sample_2, ...]
             each sample is List[List[str]]

    Flatten into a single list and deduplicate
    while preserving first-seen order.
    """
    seen = set()
    flat = []
    for struct in structs:
        for group in struct:
            for t in group:
                nt = _normalize_term_text(t)
                if not nt:
                    continue
                if nt not in seen:
                    seen.add(nt)
                    flat.append(nt)
    return flat


def _collect_maximal_terms(
    structs: List[List[List[str]]],
    is_gold: bool = True
) -> List[str]:
    """
    Collect maximal financial terms across all samples.

    For gold:
      - the first element of each group is the maximal term

    For predictions:
      - choose the longest normalized string within the group

    Returns a deduplicated flat list (order preserved).
    """
    seen = set()
    result = []

    for struct in structs:
        for group in struct:
            if not group:
                continue

            if is_gold:
                candidate = group[0]
            else:
                candidate = max(group, key=lambda x: len(_normalize_term_text(x)))

            nt = _normalize_term_text(candidate)
            if not nt:
                continue
            if nt not in seen:
                seen.add(nt)
                result.append(nt)

    return result


# ==========================
# Term HR@K & Maximal F1
# ==========================

# Currently unused, but retained for flexibility
K_VALUES = (1, 5, 10)


def term_hr_agg(items: List[Tuple[Any, Any]]) -> Dict[str, float]:
    """
    Per-sample HR@K over all terms, then averaged across samples.
    K is defined in K_VALUES.
    items: list of (gold, pred)
    """
    zeros = {f"hr_at_{k}": 0.0 for k in K_VALUES}
    if not items:
        return zeros

    # accumulate per-sample HR
    sum_hr = {f"hr_at_{k}": 0.0 for k in K_VALUES}
    valid_samples = 0

    for gold_raw, pred_raw in items:
        gold_struct = _parse_term_structure(gold_raw)
        pred_struct = _parse_term_structure(pred_raw)

        # flatten per-sample terms
        gold_terms_flat = _flatten_unique_terms([gold_struct])
        pred_terms_flat = _flatten_unique_terms([pred_struct])

        gold_term_set = set(gold_terms_flat)
        if not gold_term_set:
            # skip samples with no gold terms
            continue

        valid_samples += 1

        if not pred_terms_flat:
            # this sample contributes 0 to all hr_k
            continue

        for K in K_VALUES:
            k_eff = min(K, len(pred_terms_flat))
            pred_top_k = pred_terms_flat[:k_eff]
            pred_top_k_set = set(pred_top_k)

            hit_count = len(gold_term_set & pred_top_k_set)
            hr = hit_count / len(gold_term_set)

            sum_hr[f"hr_at_{K}"] += hr

    if valid_samples == 0:
        return zeros

    # average over valid samples and round to 4 decimals
    avg_hr = {
        f"hr_at_{K}": round(sum_hr[f"hr_at_{K}"] / valid_samples, 4)
        for K in K_VALUES
    }
    return avg_hr


def term_max_f1_agg(items: List[Tuple[Any, Any]]) -> float:
    """
    Aggregation function for maximal financial term micro-F1.
    """
    if not items:
        return 0.0

    gold_raw_list = [g for g, _ in items]
    pred_raw_list = [p for _, p in items]

    gold_structs = [_parse_term_structure(g) for g in gold_raw_list]
    pred_structs = [_parse_term_structure(p) for p in pred_raw_list]

    gold_max_terms = _collect_maximal_terms(gold_structs, is_gold=True)
    pred_max_terms = _collect_maximal_terms(pred_structs, is_gold=False)

    gold_max_set = set(gold_max_terms)
    pred_max_set = set(pred_max_terms)

    if not gold_max_set and not pred_max_set:
        return 0.0

    tp = len(gold_max_set & pred_max_set)
    precision = tp / len(pred_max_set) if pred_max_set else 0.0
    recall = tp / len(gold_max_set) if gold_max_set else 0.0

    if precision + recall == 0.0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


# ==========================
# process_results
# ==========================

def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """
    Per-sample handler for the Japanese financial term extraction task.

    doc:
      - doc["query"]   : model input text
      - doc["answer"]  : ground-truth term structure (JSON string or list-of-lists)

    results:
      - list of model outputs (we take results[0])
    """
    gold_answer = doc["answer"]
    pred_answer = results[0]

    # Both metrics share the same (gold, pred) pair
    pair = (gold_answer, pred_answer)

    return {
        "term_hr": pair,
        "term_max_f1": pair,
    }
