import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# Initialize OpenAI client with API key and base URL
client = OpenAI(api_key="sk-0a2f03c4edd74372bd81d3b5e152f707", base_url="https://api.deepseek.com")

def get_answers_from_llm(qa, max_retries=20):
    prompt = f'''{qa["input"]} 
    Ground Truth: {qa["output"]}'''
    system_message = """I need to perform data augmentation. Can you help me generate some new answers? I will provide a context, a question, and the correct answer to the question. You need to generate 5 new answers based on this information. At least one answer must be completely wrong, and the rest should be incompletely correct or synonyms of the correct answer. Because I want to save your output, your output format can only be the format in the example and cannot output other irrelevant content. Here is an example:
    Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.
    Question: Which NFL team represented the AFC at Super Bowl 50?
    Ground Truth: Denver Broncos
    Answers you produced: {"answer1": "Obama", "answer2": "Broncos", "answer3": "New England Patriots", "answer4": "Carolina Panthers", "answer5": "Taylor Swift"}
    OK, please output the answer that meets the requirements according to the following information:
    """
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.7
            )
            if response.choices:
                return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(random.uniform(10, 12))  # Add random sleep to avoid rate limiting
    return None

def process_item(item, index):
    answers = get_answers_from_llm(item)
    if answers:
        try:
            answers = json.loads(answers)
            answers['ct_qa'] = item['input']
            answers['ground truth'] = item['output']
            answers['id'] = index
            return answers
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON response: {e}")
            print(f'其 index 为 {index}, answer 为 {item}')
    return None

def main():
    with open('datasets/squad/squad_validation.json', 'r') as file:
        data = json.load(file)

    sampler = []
    total_items = max(len(data), 2000)  # Adjust if you want to process more items
    
    with ThreadPoolExecutor() as executor, tqdm(total=total_items, desc="Processing items") as pbar:
        futures = {executor.submit(process_item, item, index): index for index, item in enumerate(data[:total_items])}
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                sampler.append(result)
            pbar.update(1)

    with open('squad_sampler.json', 'w', encoding='utf-8') as f:
        json.dump(sampler, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
