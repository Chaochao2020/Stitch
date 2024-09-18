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
        
        # 使用正则表达式从预测文本中提取第一个大写字母
        match = re.search(r'[A-E]', predict_text)
        if match:
            predicted_label = match.group(0)
        else:
            predicted_label = '未匹配'  # 如果没有匹配到，设置为 None 或者其他适当的值
            print(f'匹配失败, 预测为: {predict_text} lable 为: {true_label }')
        
        # 添加到列表中
        true_answers.append(true_label)
        extracted_predictions.append(predicted_label)


accuracy = accuracy_score(true_answers, extracted_predictions)



print(f"模型准确率: {accuracy:.5f}")

