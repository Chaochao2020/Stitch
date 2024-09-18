import asyncio
import re
import fire
from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.roles.role import Role
from metagpt.schema import Message
import json
from tqdm import tqdm

class SquadAnswerSampling(Action):
    PROMPT_TEMPLATE: str = """
    Now I want to construct a question and answer data set. 
    I think you can give 10 answers according to the {instruction}. 
    Each answer is required to be different to ensure the diversity of the answers. In addition, the correctness of the answers is not required.
    Return ```json answer: {{}} ``` with NO other texts,
    your answer:
    """

    name: str = "SquadAnswerSampling"

    async def run(self, instruction: str) -> str:
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)
        return await self._aask(prompt)

    @staticmethod
    def parse_code(rsp: str) -> str:
        match = re.search(r"```json(.*)```", rsp, re.DOTALL)
        return match.group(1) if match else rsp

class SimpleCoder(Role):
    name: str = "Alice"
    profile: str = "SimpleCoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([SquadAnswerSampling])

    async def _act(self) -> Message:
        todo = self.rc.todo
        msg = self.get_memories(k=1)[0]
        code_text = await todo.run(msg.content)
        return Message(content=code_text, role=self.profile, cause_by=type(todo))

async def process_data(file_path: str) -> list:
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data

async def process_answers(data: list) -> list:
    answers = []
    role = SimpleCoder()
    for val in tqdm(data, desc="Processing", unit="item"):
        msg = f"Context: {val['context']} Question: {val['question']}" 
        # logger.info(msg)
        answer = await role.run(msg)
        json_str = re.search(r'\{.*\}', str(answer), re.DOTALL).group(0)
        answers.append(json.loads(json_str))
    return answers

def main():
    file_path = "examples/squad_samping.jsonl"
    asyncio.run(main_async(file_path))

async def main_async(file_path: str):
    data = await process_data(file_path)
    answers = await process_answers(data[100:120])
    output_file = "examples/squad_samping_output_100:120.jsonl"
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            for answer in answers:
                json.dump(answer, file)
                file.write("\n")
    except Exception as e:
        print(f"An error occurred while writing to the output file: {e}")

if __name__ == "__main__":
    fire.Fire(main)
