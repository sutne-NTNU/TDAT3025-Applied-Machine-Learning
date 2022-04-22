import random
import time

import numpy as np


class Agent:

    def __init__(self, environment):
        self.environment = environment
        self.Q_table = self.__create_QTable()

    def __create_QTable(self):
        """Creates the Q_table, initalize all values with 0

        The Q Table is three dimensional where x = row position, y = column position, this is the state, and finally
        we have z or actions for each of these states"""
        return np.zeros((self.environment.gridsize, self.environment.gridsize, len(self.environment.actions)))

    def train(self, episodes, visualizeProgress=False, ):
        # limits
        learning_rate = 0.5
        discount_rate = 0.95
        epsilon = 0.9
        min_epsilon = 0.2
        epsilon_decay = (epsilon - min_epsilon) / episodes

        # Training
        for episode in range(episodes):
            # Get new starting state
            state = self.environment.reset()

            x, y = state[0], state[1]

            reward, done = 0, False
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, len(self.environment.actions) - 1)
                else:
                    action = np.argmax(self.Q_table[x][y])
                # Move
                next_state, reward, done = self.environment.step(action)
                # Calculate new value for q_table
                old_value = self.Q_table[x][y][action]
                next_max = np.max(self.Q_table[next_state[0]][next_state[1]])
                # find new value with Bellman equation.
                new_value = old_value + learning_rate * (reward + (discount_rate * next_max) - old_value)
                self.Q_table[x][y][action] = new_value
                # set new state
                x, y = next_state[0], next_state[1]

                if visualizeProgress and episode == 10:
                    self.environment.render(self.Q_table)

            epsilon -= epsilon_decay

            if visualizeProgress and episode % 2 == 0:
                self.environment.render(self.Q_table)

    def visualize(self, numberOfVisualizations, speed=10, randomActions=0.25, timeout=4):
        """Visualize behaviour of agent witht the current state of the Q_table"""
        for i in range(numberOfVisualizations):
            state, done, timer = self.environment.reset(), False, time.time()
            while not done and time.time() - timer <= timeout:
                if random.uniform(0, 1) < randomActions:
                    action = random.randint(0, len(self.environment.actions) - 1)
                else:
                    action = np.argmax(self.Q_table[state[0]][state[1]])
                state, reward, done = self.environment.step(action)
                self.environment.render(speed=speed)
