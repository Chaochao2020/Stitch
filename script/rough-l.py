from rouge import Rouge
import json

def Rouge_L(label, predict):
    rouge = Rouge()
    scores = rouge.get_scores(predict, label)
    return scores[0]['rouge-l']['f']

with open('squad_sampler.json', 'r') as file:
    data = json.load(file)

for index, item in enumerate(data):
    ground_truth = item['ground truth']
    item['rouge_l-ground_truth'] = Rouge_L(ground_truth, ground_truth)
    item['rouge_l-1'] = Rouge_L(ground_truth, item['answer1'])
    item['rouge_l-2'] = Rouge_L(ground_truth, item['answer2'])
    item['rouge_l-3'] = Rouge_L(ground_truth, item['answer3'])
    item['rouge_l-4'] = Rouge_L(ground_truth, item['answer4'])
    item['rouge_l-5'] = Rouge_L(ground_truth, item['answer5'])

with open('squad_sampler_rouge-l.json', 'w') as file:
    json.dump(data, file, indent=4)