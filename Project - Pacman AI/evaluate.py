from agent import Agent
from environment import Environment

if __name__ == "__main__":
    # Create environment
    env = Environment()
    # Create agent in environment
    agent = Agent(env)
    # Evaluate agent to view how smart it is
    agent.evaluate(number_of_evaluations=10)
