from openai import OpenAI
from dotenv import load_dotenv
from src.reflection import InformationReflection, StrategyReflection
import os
import re

load_dotenv()

class Agent:
    def __init__(self, name, prompt):
        self.client = OpenAI(
            api_key = os.environ.get("OPENAI_API_KEY")
        )
        self.name = name
        self.prompt = prompt

    def generate_response(self, user_input):
        raise NotImplementedError("This method should be implemented by subclasses")

    def call_openai(self, prompt,user_input=None):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}] + [{"role": "user", "content": user_input}],
            temperature = 0.0
        )
        return response.choices[0].message.content


class AskingAgent(Agent):
    def generate_response(self, user_input):
        prompt = self.prompt['agents']['asking']['instruction']
        return self.call_openai(prompt,user_input)


class RecommendingAgent(Agent):
    def generate_response(self, user_input):
        prompt = self.prompt['agents']['recommending']['instruction']
        return self.call_openai(prompt,user_input)


class ChitChattingAgent(Agent):
    def generate_response(self, user_input):
        prompt = self.prompt['agents']['chit_chatting']['instruction']
        return self.call_openai(prompt,user_input)


class PlannerAgent(Agent):
    def __init__(self, name, agents,prompt):
        super().__init__(name, None)
        self.agents = agents
        self.information_reflection = InformationReflection()
        self.strategy_reflection = StrategyReflection()
        self.prompt = prompt['agents']['planner']['instruction']
        self.user_profile = []
        self.suggestions = []
        self.user_feedback =[]
        self.act_history =[]
        self.client = OpenAI(
            api_key = os.environ.get("OPENAI_API_KEY")
        )

    def plan_response(self, user_input,user_feedback,dialogue_history):
        # Collect candidate responses from all agents
        self.user_feedback.append(user_feedback)
        # Update user profile and strategy reflection
        self.user_profile = self.information_reflection.reflect(dialogue_history[-1], self.user_feedback[-1] if len(self.user_feedback)!=0 else self.user_feedback ,self.user_profile[-1] if len(self.user_profile)!=0 else self.user_profile)
        suggestions_part, experience_part = (self.strategy_reflection.reflect(dialogue_history,self.user_feedback,self.user_profile ))
        self.suggestions.append(suggestions_part)

        user_profile = self.user_profile[-1] if len(self.user_profile) !=0 else self.user_profile
        suggestions = self.suggestions[-1] if len(self.suggestions) !=0 else self.suggestions
        agents_prompt = (
            f"Dialogue history: {str(dialogue_history)}\n"
            f"User Profile: {str(user_profile)}\n"
            f"Strategy Level: {str(suggestions)}"
        )
        candidate_responses = [agent.generate_response(agents_prompt) for agent in self.agents]
        
        planner_prompt = (
            f"Dialogue history: {str(dialogue_history)}\n"
            f"Action history: {str(self.act_history)}\n"
            f"Corrective Experience: {str(experience_part)}\n"
        )
        for i, response in enumerate(candidate_responses):
            planner_prompt += f"Response {i+1}: {response}\n"
        planner_prompt += """\nBased on the user's preferences, choose the best response. 
                            You should follow this format
                            ###
                            [response number]: {number}
                            [response]: {response} """

        # Decide on the best response
        while True:
            best_response = self.call_openai(self.prompt,planner_prompt)
            
            response_number_match = re.search(r'\[response number\]: (\d+)', best_response)
            response_match = re.search(r'\[response\]: (.+)', best_response, re.DOTALL)

            # 추출된 값들을 변수에 저장
            response_number = None
            response = None

            if response_number_match:
                response_number = int(response_number_match.group(1))

            if response_match:
                response = response_match.group(1).strip()
                break
            else:
                continue

        if response_number == 1:
            self.act_history.append("ask")
        elif response_number == 2:
            self.act_history.append("recommend")
        elif response_number == 3:
            self.act_history.append("chit-chat")

        return response
