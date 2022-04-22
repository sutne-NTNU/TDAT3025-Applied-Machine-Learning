from agent import Agent
from environment import Environment

if __name__ == "__main__":
    # Create environment
    env = Environment()
    # Create agent in environment
    agent = Agent(env)
    # Train the agent for a number of episodes, evaluate routinely during training
    agent.train(number_of_episodes=16_000, evaluate_every=25)
