import json

# 文件路径
json_file_path = '/home/chao.cui/llm_finetuning/llm_multilabel_clf/paper/datasets/squad/squad_validation.json'   # 请替换为你的 json 文件路径
jsonl_file_path = '/home/chao.cui/LLaMA-Factory/saves/paper/res/Meta-Llama-3-8B-Instruct_mft_squad/generated_predictions.jsonl' # 请替换为你的 jsonl 文件路径
output_file_path = '/home/chao.cui/llm_finetuning/llm_multilabel_clf/paper/datasets/squad/Meta-Llama-3-8B-Instruct_mft_squad_merge.json' # 输出文件路径

# 读取 JSON 文件
with open(json_file_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# 读取 JSONL 文件
jsonl_data = []
with open(jsonl_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        jsonl_data.append(json.loads(line.strip()))

# 检查两个文件中的样本数量是否相同
if len(json_data) != len(jsonl_data):
    raise ValueError("JSON 文件和 JSONL 文件中的样本数量不匹配")

# 将 JSONL 文件中的 predict 添加到 JSON 文件中的每个对象
for json_obj, jsonl_obj in zip(json_data, jsonl_data):
    json_obj['predict'] = jsonl_obj['predict']

# 将合并后的数据写入输出文件
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

print("文件合并完成，输出文件路径为：", output_file_path)
