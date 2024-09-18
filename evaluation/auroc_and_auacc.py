import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def calculate_auroc_auacc_from_json(json_file, rouge_threshold=0.99):
    # 从 JSON 文件中读取数据
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # 提取打分模型的输出分数和实际 EM 标签
    scores = [item['predict'] for item in data]
    em_labels = [item['em'] for item in data]
    
    # 根据 ROUGE-L 阈值生成预测标签
    predicted_labels = np.array([1 if score >= rouge_threshold else 0 for score in scores])
    
    # 计算 AUROC
    auroc = roc_auc_score(em_labels, scores)
    
    # 计算不同阈值下的 Accuracy，并绘制 Accuracy 曲线
    accuracies = []
    thresholds = np.linspace(0, 1, 100)  # 使用多个阈值从 0 到 1
    for threshold in thresholds:
        predicted = np.array([1 if score >= threshold else 0 for score in scores])
        accuracy = accuracy_score(em_labels, predicted)
        accuracies.append(accuracy)
    
    # 计算 AUACC: 使用 Accuracy 曲线下的面积
    auacc = np.trapz(accuracies, thresholds)  # 使用梯形积分方法
    
    return auroc, auacc

# 示例输入
json_file = ''

# 调用函数计算 AUROC 和 AUACC
auroc, auacc = calculate_auroc_auacc_from_json(json_file)

print(f"AUROC: {auroc}")
print(f"AUACC: {auacc}")
