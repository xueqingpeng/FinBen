import json
import re

def normalization(text_str):
    json_str = "{\"result\": []}"
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_str, flags=re.DOTALL)
    
    if code_block_match:
        json_str = code_block_match.group(1)
    else:
        json_match = re.search(r"(\{[\s\S]*?\"result\"\s*:\s*\[.*?\][\s\S]*?\})", text_str)
        if json_match:
            json_str = json_match.group(1)

    json_str = json_str.replace("\n", "").strip()
    return json_str.strip()

def is_hashable(x):
    return isinstance(x, str)

def flatten_and_convert_to_set(data):
    if not data:
        return set()

    result = set()
    for sentence_idx, sublist in enumerate(data):
        if not sublist:
            continue
        for item in sublist:
            if item and 'Fact' in item and 'Type' in item:
                fact = item['Fact']
                typ = item['Type']
                if is_hashable(fact) and is_hashable(typ):
                    result.add((sentence_idx, fact, typ))
                else:
                    # You can log the offending items if needed
                    continue
    return result

def calculate_result(reference, prediction):
    a_set = flatten_and_convert_to_set(reference)
    b_set = flatten_and_convert_to_set(prediction)
    
    TP = len(a_set & b_set)
    FP = len(b_set - a_set)
    FN = len(a_set - b_set)

    if TP + FP == 0:
        precision = 1 if len(a_set) == 0 else 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 1
    else:
        recall = TP / (TP + FN)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def evaluate_ner(items):
    return items

def evaluate_ner_agg(items):
    true_answer = [item[0] for item in items]
    pred_answer = [item[1] for item in items]

    reference = []
    prediction = []
    for t_answer, p_answer in zip(true_answer, pred_answer):
        t_a = json.loads(t_answer)
        reference.append(t_a.get("result", []))

        p_a = normalization(p_answer)
        try:
            p_a = json.loads(p_a)
            prediction.append(p_a.get("result", []))
        except:
            prediction.append([])

    precision, recall, f1 = calculate_result(reference, prediction)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }
