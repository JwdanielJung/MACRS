import openai

class InformationReflection:
    def __init__(self):
        self.user_profile = {}  # Store user preferences and feedback

    def reflect(self, dialogue_history, user_feedback):
        prompt = (
            f"Here is the conversation so far: {dialogue_history}\n"
            "Based on this, summarize the user's preferences and create a user profile."
        )
        profile = self.call_openai(prompt)
        self.user_profile.update(profile)  # Update with new preferences
        return self.user_profile

    def call_openai(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        return response['choices'][0]['message']['content']


class StrategyReflection:
    def __init__(self):
        self.strategy = None  # Store strategy recommendations

    def reflect(self, dialogue_history, last_recommendation, user_feedback):
        prompt = (
            f"The user responded with: {user_feedback}\n"
            f"The system recommended: {last_recommendation}\n"
            f"Conversation history: {dialogue_history}\n"
            "What went wrong? How should the system adjust its strategy moving forward?"
        )
        self.strategy = self.call_openai(prompt)
        return self.strategy

    def call_openai(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
