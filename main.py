import os
import pyarrow.parquet as pq
import pandas as pd

from src.agent import AskingAgent, RecommendingAgent, ChitChattingAgent, PlannerAgent
from src.simulator import UserSimulator
from src.utils import read_yaml, read_json
from src.data.data import load_data

if __name__ == "__main__":
    # Load YAML files
    AGENTS_YAML_PATH = os.path.join("prompt", "agents.yaml")
    USERS_YAML_PATH = os.path.join("prompt", "user.yaml")
    agent_prompts = read_yaml(AGENTS_YAML_PATH)
    user_instruction = read_yaml(USERS_YAML_PATH)

    path = "/data/crs/movielens/final/240911"
    data = load_data(path)
    train = data["train"]
    movies = {
        int(k): v for k, v in read_json(os.path.join(path, "movies.json")).items()
    }
    persona_path = path + "/persona/"
    for i, row in train.iterrows():
        persona_path += f"train_{i}.json"
        persona = read_json(persona_path)

        target_item = movies[persona["movie_id"]]
        persona = persona["user_persona"]
        print(persona)
        # Create dialogue history
        dialogue_history = []

        # # Initialize agents
        asking_agent = AskingAgent("AskingAgent", agent_prompts)
        recommending_agent = RecommendingAgent("RecommendingAgent", agent_prompts)
        chitchatting_agent = ChitChattingAgent("ChitChattingAgent", agent_prompts)

        # Create planner agent
        planner_agent = PlannerAgent(
            "PlannerAgent",
            [asking_agent, recommending_agent, chitchatting_agent],
            agent_prompts,
        )

        user_instruction = (
            user_instruction["users"]["simulator_instruction"]
            + "target_movie_title: "
            + str(target_item)
        )
        # Initialize user simulator
        user_simulator = UserSimulator(
            planner_agent, user_instruction, target_item, dialogue_history
        )

        # Start the interaction
        user_simulator.interact(max_turns=5)
        break
