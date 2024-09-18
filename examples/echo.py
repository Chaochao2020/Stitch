import json
import subprocess
from tqdm import tqdm

def extract_improved_output(text, marker="完善后的回答: "):
    start_index = text.find(marker)
    if start_index == -1:
        return None  # 未找到特定子串
    start_index += len(marker)  # 跳过特定子串的长度
    return text[start_index:].strip()


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def main():
    json_file = ''  
    json_objs = read_json_file(json_file)

    for json_obj in tqdm(json_objs, desc="Processing JSON objects"):
        input_data = json_obj.get('input')
        predict_data = json_obj.get('predict')
        output_data = json_obj.get('output')
        if predict_data == output_data:
            continue
        data_to_pass = {
            'input': input_data,
            'predict': predict_data
            # 'output': output_data
        }

        result = subprocess.run(
            ['python', 'debate_comm.py'],
            input=json.dumps(data_to_pass),
            text=True,
            capture_output=True
        )
        improved_output = result.stdout.strip()
        json_obj['improved_output'] = extract_improved_output(improved_output)
        # print(f"Result for object {json_obj} ")
        # print(f'{result.stdout.strip()}')

    write_json_file(json_file, json_objs)

if __name__ == "__main__":
    main()
