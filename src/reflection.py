from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class InformationReflection:
    def __init__(self):
        self.user_profile = {}  # Store user preferences and feedback
        self.client = OpenAI(
            api_key = os.environ.get("OPENAI_API_KEY")
        )

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
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}] + [{"role": "user", "content": user_input}],
            temperature = 0.0       
            )
        return response.choices[0].message.content


class StrategyReflection:
    def __init__(self):
        self.strategy = None  # Store strategy recommendations
        self.client = OpenAI(
            api_key = os.environ.get("OPENAI_API_KEY")
        )

    def reflect(self,responses,user_feedback,user_profile):
        prompt = "Based on your past action trajectory, your goal is to write a few sentences to explain why your recommendation failed as indicated by the user utterance."
        user_input = (
            f"The user responded with: {user_feedback}\n"
            f"User Profile: {user_profile}\n"
            f"Resoibse history: {responses}\n"
        )
        error_explain = self.call_openai(prompt,user_input)
        
        prompt =(
            "You need to generate several suggestions to “Recommending Agent”, “Asking Agent” and “Chit-chatting Agent”. Then you should report the suggestions to the “Planning Agent” as experiences.\n"
            "Suggestions for Recommending Agent:\n"
            "Suggestions for Asking Agent:\n"
            "Suggestions for Chit-Chatting Agent:\n"
            "Corrective Experience for Planner Agent:\n"
        )
        recommending_agent, asking_agent, chit_chat_agent, experience_part = self.extract_suggestions_and_experience(self.call_openai(prompt,error_explain)) 
        return recommending_agent, asking_agent, chit_chat_agent, experience_part
    def extract_suggestions_and_experience(self, response_text):
        # Initialize empty strings for each suggestion and experience
        recommending_agent = ""
        asking_agent = ""
        chit_chat_agent = ""
        experience_part = ""

        # Check if the response contains all necessary sections
        if "Suggestions for Recommending Agent:" in response_text:
            recommending_agent = response_text.split("Suggestions for Recommending Agent:")[1].split("Suggestions for Asking Agent:")[0].strip()

        if "Suggestions for Asking Agent:" in response_text:
            asking_agent = response_text.split("Suggestions for Asking Agent:")[1].split("Suggestions for Chit-Chatting Agent:")[0].strip()

        if "Suggestions for Chit-Chatting Agent:" in response_text:
            chit_chat_agent = response_text.split("Suggestions for Chit-Chatting Agent:")[1].split("Corrective Experience for Planner Agent:")[0].strip()

        if "Corrective Experience for Planner Agent:" in response_text:
            experience_part = response_text.split("Corrective Experience for Planner Agent:")[1].strip()

        # Return the split sections
        return recommending_agent, asking_agent, chit_chat_agent, experience_part

    def call_openai(self, prompt,user_input):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}] + [{"role": "user", "content": user_input}],
            temperature = 0.0
        )
        response = response.choices[0].message.content
        return response