# Stitch: Enhancing Selective Prediction for Improved Performance and Reliability in LLMs

This repo contains code of the following paper:

**Stitch: Enhancing Selective Prediction for Improved Performance and Reliability in LLMs** \


## Getting Started

* **Multi-task Fine-tuning for Enhanced Performance**: Stitch fine-tunes the LLM on multiple tasks, improving its performance and generalization across different domains and tasks.

* **Model Integration for Better Output Quality:** Stitch combines a primary model with two auxiliary models, using iterative dialogue to refine outputs and significantly enhance accuracy, consistency, and overall quality.

* **RoBERTa-based Reliability Assessment:** Stitch employs a RoBERTa-based scorer to assess output reliability, without generating multiple output sequences. This approach significantly reduces computational overhead while improving performance and generalizability.

* **Reduced Risk of Erroneous Outputs:** Stitch provides a quantitative confidence indicator, enabling filtering of reliable outputs and reducing the risk of errors, especially crucial for high-risk decision-making scenarios.

  

Now, Stitch can be loaded with `transformers` as a PyTorch model.



## Requirements

Stitch's multi task fine-tuning and  model integration are both based on [MFTCoder](https://github.com/codefuse-ai/MFTCoder) and [MetaGPT](https://github.com/geekan/MetaGPT).

So please install the corresponding Python dependencies according to your needs.



| Mandatory    | Minimum | Recommend |
| ------------ | ------- | --------- |
| python       | 3.8     | 3.11      |
| torch        | 1.13.1  | 2.4.0     |
| transformers | 4.41.2  | 4.43.4    |
| datasets     | 2.16.0  | 2.20.0    |
| accelerate   | 0.30.1  | 0.32.0    |
| peft         | 0.11.1  | 0.12.0    |
| trl          | 0.8.6   | 0.9.6     |

| Optional     | Minimum | Recommend |
| ------------ | ------- | --------- |
| CUDA         | 11.6    | 12.2      |
| deepspeed    | 0.10.0  | 0.14.0    |
| bitsandbytes | 0.39.0  | 0.43.1    |
| vllm         | 0.4.3   | 0.5.0     |
| flash-attn   | 2.3.0   | 2.6.3     |



```python
import asyncio
import platform
from typing import Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import fire
import json
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team
from metagpt.config2 import Config

# You can use your own API key to prevent the current key from expiring.
moonshot_llm_config = {
    "api_type": 'moonshot',
    "base_url": 'https://api.moonshot.cn/v1',
    "api_key": 'sk-mPinXs1zjtNxtn7eJ32XYN7bbTCujxgZIDHAb9yLUReektpY',
    "model": 'moonshot-v1-8k'
}
deepseek_llm_config = {
    "api_type": 'openai',
    "api_key": 'sk-3974ca4d36f34d90bd393822efa38e93',
    "base_url": 'https://api.deepseek.com',
    "model": 'deepseek-chat'
}


QUESTION = """Which vitamin assists in blood clotting?"""
ANSWER = 'Vitamin C'
moonshot = Config.from_llm_config(moonshot_llm_config)
deepseek = Config.from_llm_config(deepseek_llm_config)
improved_answer = []
improved_output = ''



class SpeakAloud(Action):
    """Action: Speak out aloud in a debate (quarrel)"""

    PROMPT_TEMPLATE: str = """
    ## Background
    Suppose you are {name}, you and {opponent_name} will act as encyclopedias with rich knowledge, now I will give the question {question} and the answer {answer}, now you need to have a conversation with {opponent_name} based on your rich knowledge and history To improve the answer, you need to give a reasonable answer after a friendly exchange.
    ## Historical Conversations
    {context}
    ## rule:
    1. During the communication process, corresponding opinions and reasons should be given for the other party's reference.
    2. Give a reasonable answer after final negotiation. If you think the user's answer is already perfect, there is no need to improve it.
    3. Because I want to extract your output content, for the sake of uniformity, please ensure that the output format is {{"communication content": "xx", "complete answer": "xx"}}. In addition, the value of "complete answer" should be concise and powerful, and do not output irrelevant content.
    ## Related Information:
    {question}
    {answer}
    """
    name: str = "SpeakAloud"
    language: str = "English"
    question: str = QUESTION
    answer: str = ANSWER
    
    async def run(self, context: str, name: str, opponent_name: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context, name=name, opponent_name=opponent_name, language=self.language, question = self.question, answer = self.answer)
        # logger.info(prompt)

        rsp = await self._aask(prompt)

        return rsp


class Debator(Role):
    name: str = ""
    profile: str = ""
    opponent_name: str = ""

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([SpeakAloud])
        self._watch([UserRequirement, SpeakAloud])

    async def _observe(self) -> int:
        await super()._observe()
        # accept messages sent (from opponent) to self, disregard own messages from the last round
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == {self.name}]
        return len(self.rc.news)

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo  # An instance of SpeakAloud

        memories = self.get_memories()
        context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in memories)

        rsp = await todo.run(context=context, name=self.name, opponent_name=self.opponent_name)
        
        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.opponent_name,
        )
        self.rc.memory.add(msg)
        improved_answer.append(msg)
        return msg


async def debate(idea: str = "Now it's your turn to complete your answer." , investment: float = 3.0, n_round: int = 3):
    """Run a team of presidents and watch they quarrel. :)"""
    Chaochao = Debator(name="DeepSeek", profile="Democrat", opponent_name="Moonshot", config = deepseek)
    Liang = Debator(name="Moonshot", profile="Republican", opponent_name="DeepSeek", config = moonshot)
    team = Team()
    team.hire([Chaochao, Liang])
    team.invest(investment)
    team.run_project(idea, send_to="DeepSeek")  # send debate topic to Biden and let him speak first
    await team.run(n_round=n_round)
    
    

def main(idea: str = "Now it's your turn to complete your answer.", investment: float = 3.0, n_round: int = 3):
    """
    :param idea: Debate topic, such as "Topic: The U.S. should commit more in climate change fighting"
                 or "Trump: Climate change is a hoax"
    :param investment: contribute a certain dollar amount to watch the debate
    :param n_round: maximum rounds of the debate
    :return:
    """
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(debate(idea, investment, n_round))



if __name__ == "__main__":
    print(f'Question: {QUESTION}')
    print(f'Original Answer: {ANSWER}')
    print("Start to complete the answer......")

    fire.Fire(main)
    
    if "Republican: " in str(improved_answer[-1]):
        improved_output = json.loads(str(improved_answer[-1]).split("Republican: ")[1])['complete answer']
    else:
        improved_output = json.loads(str(improved_answer[-1]).split("Democrat: ")[1])['complete answer']
    print(f'Improved Answer: {improved_output}')
    print("Start rating answers......")
    tokenizer = AutoTokenizer.from_pretrained('/home/chao.cui/RoBERTa_no_sft/RoBERTa_no_sft')
    cappy = AutoModelForSequenceClassification.from_pretrained('/home/chao.cui/RoBERTa_no_sft/RoBERTa_no_sft')

    instruction = QUESTION
    response = improved_output
    inputs = tokenizer([(instruction, response), ], return_tensors='pt')
    score = cappy(**inputs).logits[0][0].item()
    print(f'Confidence: {score}')


```





Below are the scripts to recover the experiments in the paper.


## Evaluting

### Download Test Data

```shell
bash scripts/download_promptsource_test_data.sh
```

### Evaluting Stitch on Multi Task Fine-Tuning

```shell
python cappy_promptsource.py --model_name_or_path btan2/cappy-large
```

#### Results

![Stitch](img\figure1.png)


### Evaluting Stitch on Model Integration



#### Results

![Stitch](img\figure2.png)





## Acknowledgement

*Stitch* is one of the main characters in the animated film *Lilo & Stitch*. Although he was initially a powerful alien creature, Stitch later came to understand the power of love and acceptance!

![Stitch](img\Stitch.jpg)