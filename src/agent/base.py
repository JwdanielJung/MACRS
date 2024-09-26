import re
import json
from copy import deepcopy
from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    async def generate(self, **params):
        pass

    @staticmethod
    def fill_prompt(prompt, **kwargs):
        def _fill(msg, kwargs):
            all_context_input = []
            context_input = re.findall(r"(\{\{\$.+?\}\})", msg)
            for input_ in context_input:
                str_to_replace = kwargs[input_[3:-2]]
                all_context_input.append(input_[3:-2])
                if isinstance(str_to_replace, int):
                    str_to_replace = str(str_to_replace)
                if isinstance(str_to_replace, list):
                    str_to_replace = "- " + "\n- ".join(str_to_replace)
                if isinstance(str_to_replace, dict):
                    str_to_replace = json.dumps(str_to_replace).replace('", "', '",\n"')
                msg = msg.replace(input_, str_to_replace)
            return msg, all_context_input

        prompt = deepcopy(prompt)
        all_context_input = []
        if isinstance(prompt["messages"], str):
            output = _fill(prompt["messages"], kwargs)
            prompt["messages"] = output[0]
            all_context_input.extend(output[1])
        else:
            for msg in prompt["messages"]:
                output = _fill(msg["content"], kwargs)
                msg["content"] = output[0]
                all_context_input.extend(output[1])

        return prompt
