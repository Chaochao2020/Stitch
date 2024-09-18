import json
import re
from sklearn.metrics import accuracy_score

# 定义 JSONL 文件路径
jsonl_file_path = ''

# 初始化列表以存储 ground truth 和预测选项
true_answers = []
extracted_predictions = []

# 读取 JSONL 文件
with open(jsonl_file_path, 'r') as file:
    for line in file:
        # 解析每一行 JSON 对象
        data = json.loads(line)
        
        # 提取 ground truth 和预测文本
        true_label = data['label']
        predict_text = data['predict']
        
        # 根据 label 值提取预测值
        if true_label in 'ABCDE':
            # 提取第一个大写字母
            match = re.search(r'[A-E]', predict_text)
            predicted_label = match.group(0) if match else '未匹配'
        elif true_label in '1234':
            # 提取第一个数字
            match = re.search(r'[1-4]', predict_text)
            predicted_label = match.group(0) if match else '未匹配'
        else:
            predicted_label = '未匹配'
            print(f'无效的 label: {true_label}, 预测为: {predict_text}')
        
        # 添加到列表中
        true_answers.append(true_label)
        extracted_predictions.append(predicted_label)

# 计算准确率
accuracy = accuracy_score(true_answers, extracted_predictions)

print(f"模型准确率: {accuracy:.5f}")
