import json
from tqdm import tqdm
from collections import Counter
import string
import re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punctuation(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def evaluate(predictions):
    f1 = exact_match = total = 0
    for item in tqdm(predictions, desc="Evaluating"):
        ground_truth = item['label']
        prediction = item['predict']
        total += 1
        exact_match += exact_match_score(prediction, ground_truth)
        f1 += f1_score(prediction, ground_truth)
    exact_match /= total
    f1 /= total
    return {'exact_match': exact_match, 'f1': f1}

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# 加载预测结果文件
file_path = ''
predictions = load_jsonl(file_path)

# 计算并打印评测结果
results = evaluate(predictions)
print(f"EM: {results['exact_match']:.4f}, F1: {results['f1']:.4f}")
