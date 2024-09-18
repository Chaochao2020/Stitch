import json

def calculate_accuracy(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_objects = len(data)
    correct_predictions = 0
    
    for obj in data:
        output = obj['output']
        predict = obj['predict']
        
        
        if output == predict:
            correct_predictions += 1
        else:
            if 'improved_output' in obj:
                improved_output = obj['improved_output']
                if output == improved_output:
                    correct_predictions += 1
    
    # accuracy = correct_predictions / total_objects * 100
    print(correct_predictions)
    accuracy = correct_predictions / total_objects 
    return accuracy

json_file = ''
accuracy = calculate_accuracy(json_file)

print(f"Accuracy:{accuracy:.5f}")
