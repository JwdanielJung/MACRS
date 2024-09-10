import openai

class UserSimulator:
    def __init__(self, planner_agent, instruction,target_item,dialogue_history):
        self.planner_agent = planner_agent
        self.instruction = instruction
        self.target_item = target_item
        self.dialogue_history = dialogue_history
        self.accepted = False

    def call_openai(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": self.instruction + prompt}]
        )
        return response['choices'][0]['message']['content']

    def get_user_input(self, turn):
        return self.call_openai(f"Generate a response as a user in turn {turn} of a movie recommendation conversation. You must not directly mention the target information.")


    def process_feedback(self, system_response):
        #GPT사용해서 다시 feedback생성
        if str(self.target_item) in system_response:
            return True
        return False

    def interact(self, max_turns=5):
        print(f"User Simulator Instruction: {self.instruction}")
        turn = 1
        while turn <= max_turns and not self.accepted:
            #if turn == max_turns:

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
    