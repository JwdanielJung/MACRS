import openai

class UserSimulator:
    def __init__(self, planner_agent, instruction):
        self.planner_agent = planner_agent
        self.instruction = instruction
        self.accepted = False

    def call_openai(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        return response['choices'][0]['message']['content']

    def get_user_input(self, turn):
        return self.call_openai(f"Generate a response as a user in turn {turn} of a movie recommendation conversation.")

    def process_feedback(self, system_response):
        if "recommend" in system_response and "Eraser" in system_response:
            return True
        return False

    def interact(self, max_turns=5):
        print(f"User Simulator Instruction: {self.instruction['users']['simulator_instruction']}")
        turn = 1
        while turn <= max_turns and not self.accepted:
            user_input = self.get_user_input(turn)
            print(f"Turn {turn} - User: {user_input}")
            response = self.planner_agent.plan_response(user_input)
            print(f"Turn {turn} - System: {response}")
            self.accepted = self.process_feedback(response)
            turn += 1

        if self.accepted:
            print("User has accepted the recommendation.")
        else:
            print("Max turns reached, conversation ended.")
