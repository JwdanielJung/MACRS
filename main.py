import os
import pyarrow.parquet as pq
import pandas as pd

from src.agent import AskingAgent, RecommendingAgent, ChitChattingAgent, PlannerAgent
from src.simulator import UserSimulator
from src.utils import load_yaml

if __name__ == "__main__":
    # Load YAML files
    AGENTS_YAML_PATH = os.path.join("prompt", "agents.yaml")
    USERS_YAML_PATH = os.path.join("prompt", "user.yaml")
    agent_prompts = load_yaml(AGENTS_YAML_PATH)
    user_instruction = load_yaml(USERS_YAML_PATH)

    with open(os.path.join("movie_dataset","movies_add_meta.parquet"), 'rb') as file:
        data = pq.read_table(file)
    data = data.to_pandas()


    for i, row in data.iterrows():
        target_item = row['title']
        target_item_information = row[['year', 'imdb_id', 'plot_synopsis', 'tags', 'titleType', 'runtimeMinutes', 'gneres']].to_dict()

        # Create dialogue history
        dialogue_history = []

        # # Initialize agents
        asking_agent = AskingAgent("AskingAgent", agent_prompts)
        recommending_agent = RecommendingAgent("RecommendingAgent", agent_prompts)
        chitchatting_agent = ChitChattingAgent("ChitChattingAgent", agent_prompts)
        
        # Create planner agent
        planner_agent = PlannerAgent("PlannerAgent", [asking_agent, recommending_agent, chitchatting_agent],agent_prompts)
        
        user_instruction = user_instruction['users']['simulator_instruction'] + 'target_movie_title: ' + str(target_item)
        # Initialize user simulator
        user_simulator = UserSimulator(planner_agent, user_instruction,target_item,dialogue_history)

        # Start the interaction
        user_simulator.interact(max_turns=5)
        break