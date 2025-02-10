import evaluate
import re
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import json
import datetime
from collections import defaultdict
from FactScoreLite.factscore import FactScore
import jieba
import bert_score
from bert_score import BERTScorer
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel

os.environ["OPENAI_API_KEY"] = 'your api key here'

DEVICE = "cuda" if torch.cuda.is_available() else "CPU"

# Define categories
categories = ['organizations', 'legislations', 'dates', 'monetary values', 'statistics']

# Define synonyms map
SYNONYMS_MAP = {
    'emir': 'regulation (eu) no 648/2012',
    'esma': 'european securities and markets authority',
    'ccp': 'central counterparties',
    'lei': 'legal entity identifier',
    'crd': 'directive 2006/48/ec',
    'aif': 'alternative investment fund',
    'council': 'council of the european union',
    'european parliament': 'european parliament',
    'trade repositories': 'trade repository',
    'official journal': 'official journal',
    'regulation (eu) no 648/2012': 'regulation (eu) no 648/2012',
}

# bertscore
def bertscore(items):
    """
    Calculate BERTScore for a list of (reference, candidate) pairs.
    passthrough for efficiency
    """
   
    return items

def bertscore_agg(items):
    """
    Aggregate BERTScore F1 scores for a list of items.
    Higher is better.
    """

    refs = [normalize_bertscore_text(item[0]) for item in items]
    preds = [normalize_bertscore_text(item[1]) for item in items]

    # Load BERTScore metric
    bertscore_scorer = evaluate.load("evaluate-metric/bertscore",device=DEVICE)

    # Compute BERTScore
    scores = bertscore_scorer.compute(predictions=preds, references=refs, lang='en')
    
    # Use the F1 scores for aggregation
    return sum(scores["f1"]) / len(scores["f1"]) if scores["f1"] else 0.0

def normalize_bertscore_text(text):
    
    exclusions = [
        'common stock',
    ]

    ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')

    text = text.lower()

    for term in exclusions:
        term_pattern = re.compile(r'\b' + re.escape(term.lower()) + r'\b')
        text = term_pattern.sub('', text)

    # Remove stock tickers
    text = ticker_pattern.sub('', text)

    # Remove hyphens/dashes
    text = re.sub(r'[-–—]', ' ', text)

    # Remove punctuation
    text = re.sub(r'[.,\/#!$%\^&\*;:{}=\_`~()"]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# bertscore_zh
def bertscore_zh(items):
    """
    Calculate BERTScore for a list of (reference, candidate) pairs.
    passthrough for efficiency
    """
   
    return items
    
def bertscore_agg_zh(items):
    """
    Aggregate BERTScore F1 scores for a list of items.
    Higher is better.
    """

    refs = [normalize_bertscore_text_zh(item[0]) for item in items]
    preds = [normalize_bertscore_text_zh(item[1]) for item in items]

    # Load BERTScore metric
    bertscore_scorer = evaluate.load("evaluate-metric/bertscore",device=DEVICE)

    # Compute BERTScore
    scores = bertscore_scorer.compute(predictions=preds, references=refs, lang='zh')
    
    # Use the F1 scores for aggregation
    return sum(scores["f1"]) / len(scores["f1"]) if scores["f1"] else 0.0

  
def normalize_bertscore_text_zh(text):

    ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        
    # Remove stock tickers
    text = ticker_pattern.sub('', text)
    
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[.,\/#!$%\^&\*;:{}=\_`~()"]', '', text)
 
    chinese_punctuation = r'[，。！？；：、“”‘’（）【】《》]'

    # Remove chinese punctuation
    text = re.sub(chinese_punctuation, '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# rouge1
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


## exact_match

def normalize_text(text, ignore_case=False):
    cleaned_text = re.sub(r'[\n\t"\']+', '', text)
    cleaned_text = cleaned_text.lstrip()
    if "Answer:" in cleaned_text:
        cleaned_text = re.sub(r"^Answer:\s*", "", cleaned_text)
        
    if ignore_case:
        return cleaned_text.lower().strip()
    else:
        return cleaned_text.strip()

def exact_match(items):
    """
    # passthrough for efficiency
    """
    return items


def exact_match_agg(items):
    """
    Higher is better
    """
    ig_case = True
    refs = [normalize_text(item[0],ig_case) for item in items]
    preds = [normalize_text(item[1],ig_case) for item in items]
    
    exact_match = evaluate.load("exact_match")
    results = exact_match.compute(predictions=preds, references=refs)
    return results


## NER
def evaluate_ner(items):

    return items


def evaluate_ner_agg(items):
    true_answer = [item[0] for item in items]
    pred_answer = [item[1] for item in items]

    output_data = []
    for t_answer, p_answer in zip(true_answer, pred_answer):
        get_true_entities = parse_entities(t_answer)
        get_pred_entities = parse_entities(p_answer, is_generated=True)

        # print(get_true_entities)
        # print("===============")
        # print(get_pred_entities)

        for category in categories:
            get_true_set = set(get_true_entities[category])
            get_pred_set = set(get_pred_entities[category])

            f1 = calculate_metrics(get_true_set, get_pred_set)
            output_data.append(f1)
        output_data.append("")

    avg_f1 = calculate_average_metrics(output_data)
    return avg_f1


def normalize_entity(entity):
    """ Normalizes and maps entities using the synonyms map. """
    if not entity or not isinstance(entity, str):
        return ''
    
    normalized = entity.strip().lower()
    
    # Replace synonyms
    if normalized in SYNONYMS_MAP:
        normalized = SYNONYMS_MAP[normalized]
    
    # Remove punctuation-like characters
    normalized = normalized.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    
    # Handle date formats
    try:
        date_obj = datetime.datetime.strptime(normalized, "%Y-%m-%d")
        normalized = date_obj.strftime("%Y-%m-%d")
    except ValueError:
        pass
    
    return normalized


def parse_entities(entity_str, is_generated=False):
    """ Parses and normalizes entity strings. """
    entities = defaultdict(list)
    
    if not entity_str or not isinstance(entity_str, str):
        return entities
    
    try:
        if is_generated:
            entity_str = entity_str.replace("'", '"')
        
        entity_str = entity_str.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        entity_str = re.sub(r'[\n\t]+', '', entity_str)
        entity_str = re.sub(r'"""$', '', entity_str)
        entity_str = json.loads(entity_str)
        
        for key, items in entity_str.items():
            key_lower = key.lower()
            if key_lower in categories:
                if not isinstance(items, list):
                    items = [items]
                
                for item in items:
                    if isinstance(item, (str, int)):
                        normalized = normalize_entity(str(item))
                        if normalized:
                            entities[key_lower].append(normalized)
        
    except (json.JSONDecodeError, TypeError):
        pass  # Handle JSON parsing errors
    
    return entities


def calculate_metrics(gt_set, pred_set):
    """ Calculates precision, recall, and F1 score. """
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

def calculate_average_metrics(results):
    """ Calculates the average precision, recall, and F1 score across all rows. """
    total_f1, count = 0, 0
    
    for row in results:
        if row:  # Check if the row contains valid metrics
            total_f1 += float(row)
            count += 1
    
    if count == 0:
        return {"f1": 0}
    
    return {
        "f1": round(total_f1 / count, 4)
    }


## FActScore
def FActScore(items):

    return items

def FActScore_agg(items):
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]

    fact_scorer = FactScore()
    scores, _ = fact_scorer.get_factscore(generations=preds, knowledge_sources=refs)

    return scores


## Accuracy
def acc(items):
    return items


def acc_agg(items):
    true_answer = [extract_first_number(item[0]) for item in items]
    pred_answer = [extract_first_number(item[1]) for item in items]

    # Define tolerance percentage (5% allowed deviation)
    tolerance = 0.05  # 5%

    correct = 0
    for true_number, pred_number in zip(true_answer, pred_answer):
        if true_number is not None and pred_number is not None:
            difference = abs(true_number-pred_number)
            allowed_difference = true_number*tolerance

            if difference <= allowed_difference:
                correct += 1
        elif true_number is None and pred_number is None:
            correct += 1
        else:
            continue

    accuracy = correct/len(true_answer)
    return accuracy

def extract_first_number(value):
    """
    Extracts the first numeric value from a given string.
    Ignores any explanations or additional text after the number.
    Returns the number as a float or None if not found.
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    if not isinstance(value, str):
        return None
    
    match = re.search(r"-?\d+(\.\d+)?", value)
    return float(match.group(0)) if match else None 



    









