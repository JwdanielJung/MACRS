import openai
import os 
from dotenv import load_dotenv
from src.reflection import InformationReflection, StrategyReflection

class Agent:
    def __init__(self, name, prompt):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.name = name
        self.prompt = prompt

    def generate_response(self, user_input):
        raise NotImplementedError("This method should be implemented by subclasses")

    def call_openai(self, prompt,user_input=None):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}] + [{"role": "user", "content": user_input}],
        )
        return response['choices'][0]['message']['content']


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
        self.new_strategy = []
        self.user_feedback =[]

    def plan_response(self, user_input,user_feedback,dialogue_history):
        # Collect candidate responses from all agents
        self.user_feedback.append(user_feedback)
        whole_prompt = (
            f"Dialogue history: {str(dialogue_history)}\n"
            f"User Profile: {str(self.user_profile)}\n"
            f"Strategy Level: {str(self.new_strategy)}"
        )
        candidate_responses = [agent.generate_response(whole_prompt) for agent in self.agents]
        
        # Create a prompt for deciding the best response based on feedback and reflection
        feedback_prompt = (
            "You are a planning agent in a conversational recommendation system. "
            "Here are the candidate responses:\n"
        )
        for i, response in enumerate(candidate_responses):
            feedback_prompt += f"Response {i+1}: {response}\n"
        feedback_prompt += "\nBased on the user's preferences, choose the best response."

        # Decide on the best response
        best_response = self.call_openai(self.prompt,feedback_prompt)
        
        # Update user profile and strategy reflection
        self.user_profile.append(self.information_reflection.reflect(dialogue_history[-1], self.user_feedback[-1] if len(self.user_feedback)!=0 else self.user_feedback ,self.user_profile[-1] if len(self.user_profile)!=0 else self.user_profile))
        self.new_strategy.append(self.strategy_reflection.reflect(dialogue_history,self.user_feedback,self.user_profile ))
        return best_response
