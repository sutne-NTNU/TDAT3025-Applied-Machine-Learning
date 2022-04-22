import matplotlib.pyplot as plt
import numpy as np
import time
import math


# Code is inspired by RJBooker´s solution to the cartpole problem, see README.md for links and further info


class Agent:

    def __init__(self, env):
        self.Q_table = np.zeros(env.n_bins + (env.action_space.n,))
        self.env = env
        self.scores = []

    def visualize(self, duration=5, attemptDuration=5):
        """Visualize the agent in the environment for the set amount of seconds"""
        start_visalization = time.time()
        while time.time() - start_visalization <= duration:
            state, done, start_attempt = self.env.reset(), False, time.time()
            while time.time() - start_attempt <= attemptDuration:
                # policy action
                action = self.nextAction(state)
                # increment enviroment
                state, _, done = self.env.step(action)
                self.env.render()

    def nextAction(self, state, min_ExplorationRate=0.05, episodeNr=-1):
        """Returns the new action to take based on the current state"""
        if not episodeNr < 0 and np.random.random() < self.exploration_rate(episodeNr, min_ExplorationRate):
            # explore actions
            return self.env.action_space.sample()
        else:
            # follow optimal learnt value
            return np.argmax(self.Q_table[state])

    @staticmethod
    def exploration_rate(episodeNr, min_rate):
        """Decaying exploration rate"""
        return max(min_rate, min(1.0, 1.0 - math.log10((episodeNr + 1) / 25)))

    @staticmethod
    def learning_rate(episodeNr, min_rate):
        """Decaying learning rate"""
        return max(min_rate, min(1.0, 1.0 - math.log10((episodeNr + 1) / 25)))

    def train(self, min_LearningRate=0.05, min_ExplorationRate=0.1, visualizeProgress=False, episodeLimit=300):
        episodeNr = 0
        while not self.env.solved(self.scores) and episodeNr <= episodeLimit:
            # get starting state
            state = self.env.reset()
            done, score = False, 0
            while not done:
                # find next action from current state
                action = self.nextAction(state, min_ExplorationRate, episodeNr)
                # perform the action and get the results
                next_state, reward, done = self.env.step(action)
                score += reward
                # Update Q-Table
                lr = self.learning_rate(episodeNr, min_LearningRate)
                self.Q_table[state][action] = self.__new_Q_value(state, action, reward, next_state, lr)
                # move to new state
                state = next_state
                # Render the cartpole environment
                if visualizeProgress and episodeNr % 10 == 0:
                    self.env.render()

            self.scores.append(score)
            episodeNr += 1

        if episodeNr < episodeLimit:
            print("Solved after %d episodes" % (episodeNr - 100))  # last 100 are used to verify agent is stable
        else:
            print("Failed to solve in less than %d episodes" % episodeLimit)

    def __new_Q_value(self, state, action, reward, next_state, lr, discount_factor=1.0):
        """Temperal diffrence for updating Q-value of state-action pair"""
        old_value = self.Q_table[state][action]
        future_optimal_value = np.max(self.Q_table[next_state])
        new_value = (1 - lr) * old_value + lr * (reward + discount_factor * future_optimal_value)
        # find new value with Bellman equation.
        # new_value = old_value + lr * (reward + (discount_factor * future_optimal_value) - old_value)
        return new_value

    def plotScores(self, save=False):
        """Plot the scores of each episode from the training, with the option to save the plot"""
        plt.plot(self.scores, 'r.')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward/Steps')
        plt.show()
        if save:
            plt.savefig("./media/CartPole.png")
