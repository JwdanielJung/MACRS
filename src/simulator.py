from openai import OpenAI
from dotenv import load_dotenv
import os
from src.agent.llm import OpenAIAgent

load_dotenv()


class UserSimulator:
    def __init__(self, target_item, target_movie_information,datasettype,recommend_multiple):
        # # # Initialize agents
        self._datasettype = datasettype
        self._recommend_multiple = recommend_multiple     
        self._asking_agent_prompt = "/home/ubuntu/taeseung/MACRS/prompt/ask.yaml"
        self._recommending_agent_prompt = (
            "/home/ubuntu/taeseung/MACRS/prompt/recommend.yaml"
        )
        self._chitchatting_agent_prompt = (
            "/home/ubuntu/taeseung/MACRS/prompt/chit_chat.yaml"
        )
        if self._recommend_multiple:
            self._planning_agent_prompt = "/home/ubuntu/taeseung/MACRS/prompt/planner.yaml"
        else:
            self._planning_agent_prompt = "/home/ubuntu/taeseung/MACRS/prompt/single_recommend_planner.yaml"
        if self._datasettype =="persona":
            self._user_prompt = "/home/ubuntu/taeseung/MACRS/prompt/user_persona.yaml"
        else:
            self._user_prompt = "/home/ubuntu/taeseung/MACRS/prompt/user_MACRS.yaml"
        self._fallback_prompt = "/home/ubuntu/taeseung/MACRS/prompt/fallback.yaml"
        self._info_reflection_prompt = (
            "/home/ubuntu/taeseung/MACRS/prompt/info_reflection.yaml"
        )
        self._experience_prompt = "/home/ubuntu/taeseung/MACRS/prompt/experience.yaml"
        self._strategy_level_prompt = (
            "/home/ubuntu/taeseung/MACRS/prompt/strategy_level.yaml"
        )
        self._target_item = target_item
        if self._datasettype =="persona":
            self._persona = target_movie_information["user_persona"]
            self._tags = target_movie_information["top_k_tags"]
        else:
            self.target_movie_information = target_movie_information
        self.dialogue_history = []
        self.accepted = False
        self._agent = OpenAIAgent(api_key=os.getenv("OPENAI_API_KEY"))
        self.user_profile = []
        self.ask_strategy_level = ""
        self.chit_chat_strategy_level = ""
        self.recommend_strategy_level = ""
        self.act_history = []
        self.experience = ""
        self.trajectory = []
        self.indicator = 0
        self.error = ""
        self.turn = 1

    def get_user_input(self, turn):
        previous_response = (
            self.dialogue_history[-1]
            if len(self.dialogue_history) != 0
            else self.dialogue_history
        )
        return self.call_openai(
            f"Create a user inquiry for a movie recommendation based on the context provided in {previous_response}. Avoid replicating {previous_response} word for word and refrain from explicitly naming the target movie. Try not to be a recommender system"
        )

    def _conversation_to_string(self, conversation):
        return "\n".join(
            [
                f"{'seeker'if message['role'] == 'user' else 'recommender'}: {message['content']}"
                for message in conversation
            ]
        )

    def process_feedback(self, system_response):
        if str(self._target_item) in str(system_response):
            return True
        return False

    def extract_content(self, text):
        lines = text.split("\n")
        result = {}
        current_key = None

        for line in lines:
            if line.startswith("- "):
                parts = line[2:].split(": ", 1)
                if len(parts) == 2:
                    current_key = parts[0]
                    result[current_key] = parts[1]
            elif current_key and line.strip():
                result[current_key] += " " + line.strip()

        return result

    async def generate_system_respone(self, max_turns):
        if (self.turn == max_turns) and not self.accepted:
            final_response = await self._agent.generate(
                self._fallback_prompt,
                conversation=self._conversation_to_string(self.dialogue_history),
            )
            self.act_history.append("fall_back")
        else:
            ask_response = await self._agent.generate(
                self._asking_agent_prompt,
                conversation=self._conversation_to_string(self.dialogue_history),
                user_profile=self.user_profile[-1:],
                strategy_level=self.ask_strategy_level,
            )
            chit_chat_response = await self._agent.generate(
                self._chitchatting_agent_prompt,
                conversation=self._conversation_to_string(self.dialogue_history),
                user_profile=self.user_profile[-1:],
                strategy_level=self.chit_chat_strategy_level,
            )
            if self._recommend_multiple:
                recommend_response = await self._agent.generate(
                    self._recommending_agent_prompt,
                    conversation=self._conversation_to_string(self.dialogue_history),
                    user_profile=self.user_profile[-1:],
                    strategy_level=self.recommend_strategy_level,
                )
                planner_response = await self._agent.generate(
                self._planning_agent_prompt,
                conversation=self._conversation_to_string(self.dialogue_history),
                act_history=self.act_history,
                ask=ask_response,
                chitchat=chit_chat_response,
                recommend=recommend_response,
                experience=self.experience,
            )
            else:
                planner_response = await self._agent.generate(
                self._planning_agent_prompt,
                conversation=self._conversation_to_string(self.dialogue_history),
                act_history=self.act_history,
                ask=ask_response,
                chitchat=chit_chat_response,
                experience=self.experience,
            )
            if ":" in planner_response:
                split_words = planner_response.split(":")
                self.act_history.append(split_words[0])
                final_response = split_words[-1]
            else:
                split_words = [planner_response]  # or handle it however you prefer
                final_response = split_words[-1]
        print(self.act_history)
        return final_response
    
    async def generate_user_response(self):
        if self._datasettype == "persona":
            return await self._agent.generate(
                    self._user_prompt,
                    history=self.dialogue_history,
                    reverse_role=True,
                    tags=self._tags,
                    user_persona=self._persona,
            )
        else:
            return await self._agent.generate(
                    self._user_prompt,
                    history=self.dialogue_history,
                    reverse_role=True,
                    target_information = self.target_movie_information
            )

    async def interact(self, max_turns=5):
        print(f"Simulator starts")
        while self.turn <= max_turns and not self.accepted:
            user_input = await self.generate_user_response()
            print(f"Turn {self.turn} - User: {user_input}")

            self.dialogue_history.append({"role": "user", "content": user_input})

            # can be exchanged with other recommender to get response
            final_response = await self.generate_system_respone(max_turns)

            # Append the selected response to the dialogue history
            self.dialogue_history.append(
                {"role": "assistant", "content": final_response}
            )
            print(f"Turn {self.turn} - System: {final_response}")
            self.accepted = self.process_feedback(final_response)

            user_profile = await self._agent.generate(
                self._info_reflection_prompt,
                prev_dialog=self._conversation_to_string(self.dialogue_history[-2:]),
                user_profile=self.user_profile[-1:],
            )
            self.user_profile.append(user_profile)
            self.trajectory.append([self.dialogue_history[-2], user_profile])
            if self.accepted:
                print("User has accepted the recommendation.")
                break
            else:
                if (
                    "recommend " == self.act_history[-1]
                    or "recommender" == self.act_history[-1]
                ):
                    self.experience = await self._agent.generate(
                        self._experience_prompt,
                        trajectory=str(self.trajectory[self.indicator :]),
                    )
                    # need to divide into strategy level for each agent and experience for planning agnet
                    if self._recommend_multiple:
                        self.error = await self._agent.generate(
                            self._strategy_level_prompt,
                            trajectory=str(self.trajectory[self.indicator :]),
                        )
                    else:
                        self.error = await self._agent.generate(
                            self._strategy_level_prompt,
                            trajectory=str(self.trajectory[self.indicator :]),
                        )
                    # need to divide into strategy level for each agent and experience for planning agnet
                    strategy = self.extract_content(self.error)
                    self.ask_strategy_level = strategy["ask"]
                    self.chit_chat_strategy_level = strategy["chit-chat"]
                    if self._recommend_multiple:
                        self.recommend_strategy_level = strategy["recommend"]
                    # indicator changes
                    self.indicator = self.turn
            self.turn += 1

        print("conversation ended")
