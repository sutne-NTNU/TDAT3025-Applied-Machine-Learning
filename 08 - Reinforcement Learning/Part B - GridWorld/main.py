import time

from environment_gridworld import Environment
from agent_gridworld import Agent

if __name__ == "__main__":
    # Create the environment with gridsize
    environment = Environment(gridsize=25,
                              obsticlePercentage=0.10,
                              bonusPercentage=0.02,
                              dangerPercentage=0.02)
    # Create an agent for this environment
    agent = Agent(environment)

    # Visualize agent before training
    agent.visualize(numberOfVisualizations=6, timeout=5)

    # Train the agent, visualize how Q-table changes for each episode
    agent.train(episodes=1000, visualizeProgress=True)

    # Finally visualize the agent after training
    agent.visualize(numberOfVisualizations=25, randomActions=0.0)
