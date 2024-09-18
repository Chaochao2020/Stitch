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
        # print(data['predict'])
        # predict_text = data['predict'].strip()
        predict_text = data['predict'][0]
        # print(predict_text)
         
       
        # 添加到列表中
        true_answers.append(true_label)
        extracted_predictions.append(predict_text)


accuracy = accuracy_score(true_answers, extracted_predictions)



print(f"模型准确率: {accuracy:.5f}")