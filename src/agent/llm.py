from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from src.agent.base import Agent
from src.utils import read_yaml
import logging

load_dotenv()


class OpenAIAgent(Agent):
    def __init__(self, api_key=None):
        self._client = AsyncOpenAI(api_key=api_key)

    async def generate(
        self, prompt, history=[], reverse_role=False, get_usage=False, **params
    ):
        if prompt[-5:] == ".yaml":
            prompt = read_yaml(prompt)
            if history != []:
                if prompt["messages"][0]["role"] == "system":
                    prompt["messages"] = (
                        prompt["messages"][:1]
                        + [
                            {"role": hist["role"], "content": hist["content"]}
                            for hist in history
                        ]
                        + prompt["messages"][1:]
                    )
                else:
                    prompt["messages"] = [
                        {"role": hist["role"], "content": hist["content"]}
                        for hist in history
                    ] + prompt["messages"]

        if reverse_role:
            prompt["messages"] = [
                {
                    "role": "assistant" if message["role"] == "user" else "user",
                    "content": message["content"],
                }
                for message in prompt["messages"]
            ]
        filled_prompt = self.fill_prompt(prompt, **params)
        response = await self._client.chat.completions.create(**filled_prompt)

        if get_usage:
            usage = dict(response.usage)
            return [choice.message.content for choice in response.choices][0], usage
        return [choice.message.content for choice in response.choices][0]


# class AnthropicAgent(Agent):
#     def __init__(self, api_key=None):
#         self.client = AsyncAnthropic(api_key=api_key, max_retries=0)

#     async def generate(self, **params):
#         response = await self.client.messages.create(**params)
#         # usage = dict(response.usage)
#         return response.content[0].text
