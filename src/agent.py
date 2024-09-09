import openai
from src.reflection import InformationReflection, StrategyReflection

class Agent:
    def __init__(self, name, dialogue_history, prompt):
        self.name = name
        self.dialogue_history = dialogue_history
        self.prompt = prompt

    def generate_response(self, user_input):
        raise NotImplementedError("This method should be implemented by subclasses")

    def call_openai(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}] + [{"role": "user", "content": user_input}],
        )
        return response['choices'][0]['message']['content']


class AskingAgent(Agent):
    def generate_response(self, user_input):
        prompt = self.prompt['agents']['asking']['instruction']
        return self.call_openai(prompt)


class RecommendingAgent(Agent):
    def generate_response(self, user_input):
        prompt = self.prompt['agents']['recommending']['instruction']
        return self.call_openai(prompt)


class ChitChattingAgent(Agent):
    def generate_response(self, user_input):
        prompt = self.prompt['agents']['chit_chatting']['instruction']
        return self.call_openai(prompt)


class PlannerAgent(Agent):
    def __init__(self, name, dialogue_history, agents):
        super().__init__(name, dialogue_history, None)
        self.agents = agents
        self.information_reflection = InformationReflection()
        self.strategy_reflection = StrategyReflection()

    def plan_response(self, user_input):
        # Collect candidate responses from all agents
        candidate_responses = [agent.generate_response(user_input) for agent in self.agents]
        
        # Create a prompt for deciding the best response based on feedback and reflection
        feedback_prompt = (
            "You are a planning agent in a conversational recommendation system. "
            "Here are the candidate responses:\n"
        )
        for i, response in enumerate(candidate_responses):
            feedback_prompt += f"Response {i+1}: {response}\n"
        feedback_prompt += "\nBased on the user's preferences, choose the best response."

        # Decide on the best response
        best_response = self.call_openai(feedback_prompt)
        
        # Update user profile and strategy reflection
        user_profile = self.information_reflection.reflect(self.dialogue_history, user_input)
        new_strategy = self.strategy_reflection.reflect(self.dialogue_history, best_response, user_input)
        
        # Append the selected response to the dialogue history
        self.dialogue_history.append(best_response)
        return best_response
