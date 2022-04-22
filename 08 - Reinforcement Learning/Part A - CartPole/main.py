from agent_cartpole import Agent
from environment_cartpole import Environment


def main():
    # create environment and the agent
    environment = Environment('CartPole-v0', n_bins=(6, 12))
    agent = Agent(environment)

    print("visualizing agent with no training")
    agent.visualize(duration=5, attemptDuration=1)

    print("Training agent until it solves the enivronment")
    agent.train(min_LearningRate=0.05, min_ExplorationRate=0.1, visualizeProgress=False)

    print("Visualizing agent after training")
    agent.visualize(duration=20, attemptDuration=20)

    agent.plotScores(save=False)


if __name__ == "__main__":
    main()
