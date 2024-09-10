import openai

class InformationReflection:
    def __init__(self):
        self.user_profile = {}  # Store user preferences and feedback

    def reflect(self ,dialogue_history,user_feedback,user_profile):
        prompt = "Please infer user preferences based on the conversation. And combine them with the past preferences to summarize a more complete user preferences."
        user_input = (
            f"Here is the conversation so far: {dialogue_history}\n"
            f"Previous user profile:{user_profile}\n"
            "Based on this, summarize the user's preferences and create a user profile."
        )
        profile = self.call_openai(prompt,user_input)
        # self.user_profile.update(profile)  # Update with new preferences
        return self.user_profile

    def call_openai(self, prompt,user_input):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}] + [{"role": "user", "content": user_input}],        )
        return response['choices'][0]['message']['content']


class StrategyReflection:
    def __init__(self):
        self.strategy = None  # Store strategy recommendations

    def reflect(self,dialogue_history,user_feedback,user_profile):
        prompt = "Based on your past action trajectory, your goal is to write a few sentences to explain why your recommendation failed as indicated by the user utterance."
        user_input = (
            f"The user responded with: {user_feedback}\n"
            f"User Profile: {user_profile}\n"
            f"Conversation history: {dialogue_history}\n"
            "What went wrong? How should the system adjust its strategy moving forward?"
        )
        self.strategy = self.call_openai(prompt,user_input)
        return self.strategy

    def call_openai(self, prompt,user_input):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}] + [{"role": "user", "content": user_input}],
        )
        return response['choices'][0]['message']['content']
