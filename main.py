import os
from src.agent import AskingAgent, RecommendingAgent, ChitChattingAgent, PlannerAgent
from src.simulator import UserSimulator
from src.utils import load_yaml

if __name__ == "__main__":
    # Load YAML files
    AGENTS_YAML_PATH = os.path.join("prompt", "agents.yaml")
    USERS_YAML_PATH = os.path.join("prompt", "users.yaml")
    agent_prompts = load_yaml(AGENTS_YAML_PATH)
    user_instruction = load_yaml(USERS_YAML_PATH)

    # Create dialogue history
    dialogue_history = []

    # Initialize agents
    asking_agent = AskingAgent("AskingAgent", dialogue_history, agent_prompts)
    recommending_agent = RecommendingAgent("RecommendingAgent", dialogue_history, agent_prompts)
    chitchatting_agent = ChitChattingAgent("ChitChattingAgent", dialogue_history, agent_prompts)

    # Create planner agent
    planner_agent = PlannerAgent("PlannerAgent", dialogue_history, [asking_agent, recommending_agent, chitchatting_agent])

    # Initialize user simulator
    user_simulator = UserSimulator(planner_agent, user_instruction)

    # Start the interaction
    user_simulator.interact(max_turns=5)
