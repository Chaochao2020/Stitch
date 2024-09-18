"""
Filename: MetaGPT/examples/debate.py
Created Date: Tuesday, September 19th 2023, 6:52:25 pm
Author: garylin2099
@Modified By: mashenquan, 2023-11-1. In accordance with Chapter 2.1.3 of RFC 116, modify the data type of the `send_to`
        value of the `Message` object; modify the argument type of `get_by_actions`.
"""

import asyncio
import platform
from typing import Any

import fire

from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team
from metagpt.config2 import Config


glm_llm_config = {
    "api_type": 'zhipuai',
    "api_key": '07642a3740ecf4ce52262d7c054512f2.I1ppqSaHwMxVSi56',
    "base_url": 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
    "model": 'glm-3-turbo'

}
yi_llm_config = {
    "api_type": 'yi',
    "base_url": 'https://api.lingyiwanwu.com/v1',
    "api_key": '9fea7b76c9a14107b0fde6c1b3677687',
    "model": 'yi-34b-chat-0205',
    "max_token": 4000
}

moonshot_llm_config = {
    "api_type": 'moonshot',
    "base_url": 'https://api.moonshot.cn/v1',
    "api_key": 'sk-ILSUSX5S7Dy6ob9binrcEOZKqgTXarqv17VzOOMQKc73RUWq',
    "model": 'moonshot-v1-8k'
}
deepseek_llm_config = {
    "api_type": 'openai',
    "api_key": 'sk-3974ca4d36f34d90bd393822efa38e93',
    "base_url": 'https://api.deepseek.com',
    "model": 'deepseek-chat'
}

glm = Config.from_llm_config(glm_llm_config)
yi = Config.from_llm_config(yi_llm_config)
moonshot = Config.from_llm_config(moonshot_llm_config)
deepseek = Config.from_llm_config(deepseek_llm_config)





class SpeakAloud(Action):
    """Action: Speak out aloud in a debate (quarrel)"""

    PROMPT_TEMPLATE: str = """
    ## 背景
    假定你是 {name}, 你和 {opponent_name} 将作为知识丰富的专家，现在有一个用户给出了关于问题:{question}的答案 {answer}，现在你们两个要根据这些条件以及历史对话来完善用户的答案，你们在友好交流后后最终给出一个合理的答案。
    ## 历史对话
    {context}
    ## 规则:
    1. 在交流过程中要给出相应的观点和理由以供对方参考。
    1. 输出语言为 {language}.
    2. 最终协商好后给出一个合理答案，如果认为用户的答案已经很完美则不需要再进行完善。
    3. 最终只需要输出完善的答案即可。
    ## 相关信息:
    {question}
    {answer}
    """
    name: str = "SpeakAloud"
    language: str = "English"
    question: str = "Context: Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary. Question: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
    answer: str = 'Donghua university' 
    
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
        # print(context)

        rsp = await todo.run(context=context, name=self.name, opponent_name=self.opponent_name)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.opponent_name,
        )
        self.rc.memory.add(msg)

        return msg


async def debate(idea: str = '现在轮到你来完善答案了' , investment: float = 3.0, n_round: int = 12):
    """Run a team of presidents and watch they quarrel. :)"""
    Chaochao = Debator(name="deepseek", profile="Democrat", opponent_name="月之暗面", config = deepseek)
    Liang = Debator(name="月之暗面", profile="Republican", opponent_name="deepseek", config = moonshot)
    team = Team()
    team.hire([Chaochao, Liang])
    team.invest(investment)
    team.run_project(idea, send_to="deepseek")  # send debate topic to Biden and let him speak first
    await team.run(n_round=n_round)


def main(idea: str = '现在轮到你来完善答案了', investment: float = 3.0, n_round: int = 4):
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
    fire.Fire(main)  # run as python debate.py --idea="TOPIC" --investment=3.0 --n_round=5
