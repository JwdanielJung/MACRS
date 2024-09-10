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
        target_item_information ={}
        target_item_information['year'] = row['year']
        # target_item_information['imdb_id']= row['imdb_id']
        target_item_information['plot_synopsis'] = row['plot_synopsis']
        target_item_information['tags'] = row['tags']
        target_item_information['titleType'] = row['titleType']
        target_item_information['runtimeMinutes'] = row['runtimeMinutes']
        target_item_information['gneres'] = row['gneres']

        # Create dialogue history
        dialogue_history = []

        # # Initialize agents
        asking_agent = AskingAgent("AskingAgent", agent_prompts)
        recommending_agent = RecommendingAgent("RecommendingAgent", agent_prompts)
        chitchatting_agent = ChitChattingAgent("ChitChattingAgent", agent_prompts)
        
        # Create planner agent
        planner_agent = PlannerAgent("PlannerAgent", [asking_agent, recommending_agent, chitchatting_agent],agent_prompts)
        
        user_instruction = user_instruction['users']['simulator_instruction'] +str(target_item)
        # +"\nTarget_item_information: "+str(target_item_information)
        # Initialize user simulator
        user_simulator = UserSimulator(planner_agent, user_instruction,target_item,dialogue_history)

        # Start the interaction
        user_simulator.interact(max_turns=5)
        break
        # evaluation metric
        