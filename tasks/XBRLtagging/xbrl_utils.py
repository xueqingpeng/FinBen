import re
import json

# def normalization(text_str):
#     clean_text = re.sub(r"```(?:json)?\s*\n|\n```", "", text_str, flags=re.MULTILINE)
#     clean_text = clean_text.replace("\n", "").strip()
#     return clean_text.strip()

def normalization(text_str):
    # Try to match JSON content wrapped by ```json or ```
    json_str = "{\"result\": []}"
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_str, flags=re.DOTALL)
    
    if code_block_match:
        json_str = code_block_match.group(1)
    else:
        # If there is no code block format, try to match the structure of {"result": [...]}
        json_match = re.search(r"(\{[\s\S]*?\"result\"\s*:\s*\[.*?\][\s\S]*?\})", text_str)
        if json_match:
            json_str = json_match.group(1)

    json_str = json_str.replace("\n", "").strip()
    return json_str.strip()


# def flatten_and_convert_to_set(data):
#     if not data:
#         return set()
        
#     return set(
#         (item['Fact'], item['Type'])
#         for sublist in data if sublist
#         for item in sublist if item and 'Fact' in item and 'Type' in item
#     )

def flatten_and_convert_to_set(data):
    if not data:
        return set()

    return set(
        (sentence_idx, item['Fact'], item['Type'])
        for sentence_idx, sublist in enumerate(data) if sublist
        for item in sublist if item and 'Fact' in item and 'Type' in item
    )


def calculate_result(reference, prediction):
    # flatten reference and prediction
    a_set = flatten_and_convert_to_set(reference)
    b_set = flatten_and_convert_to_set(prediction)
    
    # count True Positive (TP), False Positive (FP), False Negative (FN)
    TP = len(a_set & b_set)  # Intersection
    FP = len(b_set - a_set)  # The prediction is present but the reference is not present
    FN = len(a_set - b_set)  # The reference is present but the prediction is not present.
    
    # calculate Precision, Recall, F1
    # precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    # recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    # f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

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
    """ pass the parameters"""

    return items


def evaluate_ner_agg(items):
   
    true_answer = [item[0] for item in items]
    pred_answer = [item[1] for item in items]

    reference = []
    prediction = []
    for t_answer, p_answer in zip(true_answer, pred_answer):
       
        t_a = json.loads(t_answer)
        t_a = t_a.get("result")
        reference.append(t_a)

        p_a = normalization(p_answer)
        try:
            p_a = json.loads(p_a)
            p_a = p_a.get("result")
        except:
            p_a = []
        prediction.append(p_a)
    
    precision, recall, f1 = calculate_result(reference, prediction)
    
    
    return {"precision": round(precision, 4), 
            "recall": round(recall, 4), 
            "f1": round(f1, 4)}
