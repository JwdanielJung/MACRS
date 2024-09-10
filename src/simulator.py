from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class UserSimulator:
    def __init__(self, planner_agent, instruction,target_item,dialogue_history):
        self.planner_agent = planner_agent
        self.instruction = instruction
        self.target_item = target_item
        self.dialogue_history = dialogue_history
        self.accepted = False
        self.client = OpenAI(
            api_key = os.environ.get("OPENAI_API_KEY")
        )

    def call_openai(self, prompt):
        # print(self.instruction)
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": self.instruction + prompt}]
        )
        return response.choices[0].message.content

    def get_user_input(self, turn):
        previous_response = self.dialogue_history[-1] if len(self.dialogue_history)!=0 else self.dialogue_history
        return self.call_openai(f"Create a user inquiry for a movie recommendation based on the context provided in {previous_response}. Avoid replicating {previous_response} word for word and refrain from explicitly naming the target movie.")

    def process_feedback(self, system_response):
        if str(self.target_item) in str(system_response):
            return True
        return False

    def interact(self, max_turns=5):
        print(f"User Simulator Instruction: {self.instruction}")
        turn = 1
        while turn <= max_turns and not self.accepted:
            user_input = self.get_user_input(turn)
            print(f"Turn {turn} - User: {user_input}")
            self.dialogue_history.append(user_input)
            response = self.planner_agent.plan_response(user_input,self.accepted,self.dialogue_history)
            # Append the selected response to the dialogue history
            self.dialogue_history.append(response)
            print(f"Turn {turn} - System: {response}")
            self.accepted = self.process_feedback(response)
            turn += 1

        if self.accepted:
            print("User has accepted the recommendation.")
        else:
            print("Max turns reached, conversation ended.")
    