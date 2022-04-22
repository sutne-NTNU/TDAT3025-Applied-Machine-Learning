import random
from collections import namedtuple

import torch.nn.functional as F
import torch.optim as optim

from data_util import *
from dqn import DQN
from environment_manager import EnvironmentManager
from epsilon_greedy_strategy import EpsilonGreedyStrategy
from q_values import QValues
from replay_memory import ReplayMemory

batch_size = 32
gamma = 0.99
eps_start = 0.37
eps_end = 0.15
eps_decay = 0.000002  # per total step
eps_evaluation = 1.0
learning_rate = 0.0001
target_update_every = 5_000  # steps
memory_size = 10_000

# we store the machines experience in our replay memory
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    return t1, t2, t3, t4


class Agent:
    """
    Class over an agent in the environment
    """

    def __init__(self, environment):
        # Find device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Handle the environment
        self.em = EnvironmentManager(environment, self.device)
        self.num_actions = self.em.num_actions_available()
        self.strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
        # Create brain
        self.memory = ReplayMemory(memory_size)
        self.policy_network = self.load_network('policy_network')
        self.target_network = self.load_network('target_network')
        # Keep track of how many steps we have taken for decreasing exploration rate
        self.current_step = 0
        self.numberOfTrainedEpisodes = getLineCount(learning_values)

    def load_network(self, name):
        # if we have saved networks load them
        network = DQN().to(self.device)
        savedStateDict = loadStateDict(name)
        if savedStateDict is not None:
            network.load_state_dict(savedStateDict)
        elif name == 'target_network':
            network.load_state_dict(self.policy_network.state_dict())
        return network

    def train(self, number_of_episodes, evaluate_every, plot_results=True):
        # self.target_network.train()
        steps_and_scores, step_count = [], 0

        for episode in range(number_of_episodes):
            # Evaluate current state of agent
            if episode % evaluate_every == 0:
                self.evaluate(save_values=True)

            self.numberOfTrainedEpisodes += 1
            print("Episode", episode + 1, ":", end='')
            step_count = 0

            state = self.em.reset()  # resets environment

            while True:
                step_count += 1
                state = self.perform_action(state)
                self.optimize()

                # Stop when done
                if self.em.done:
                    score = self.em.get_score()
                    print(' \tScore:', score)
                    steps_and_scores.append((step_count, score))
                    break

            # Update target network
            if episode % target_update_every == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

            # Save networks and scores to file every 25 episodes, and when done
            if episode % 25 == 0 or episode == number_of_episodes - 1:
                save_to_file(learning_values, steps_and_scores)
                saveStateDict(self.policy_network.state_dict(), 'policy_network')
                saveStateDict(self.target_network.state_dict(), 'target_network')
                steps_and_scores = []

        if plot_results:
            plot_learning_values()
            plot_evaluation_values()

    def evaluate(self, number_of_evaluations=1, save_values=False):
        # self.target_network.eval()  # not in training mode
        episodesTrained_scores = []
        for evaluation in range(number_of_evaluations):
            print("\t\t\t\tEvaluation score:", end='')
            state = self.em.reset()

            while True:
                state = self.perform_action(state, evaluate=True)
                # Stop when done
                if self.em.done:
                    score = self.em.get_score()
                    print('\t', score)
                    episodesTrained_scores.append((self.numberOfTrainedEpisodes, score))
                    break

        if save_values:
            save_to_file(evaluation_values, episodesTrained_scores)

    def perform_action(self, state, evaluate=False):
        """
            - For each step in the episode we have to choose an action to take so we can move around the map.
            - Depending on the action taken we receive a reward which the agent uses to get better since it wants to
              maximize its reward.
            - We then get the next state so the agent can continue playing.
            - This is then pushed to our replay memory which is sampled to choose which action to take in the future.
            - If our agent does something to get killed we stop the episode.
        """
        action = self.select_action(state, evaluate)
        reward = self.em.take_action(action)
        next_state = self.em.get_state()
        self.memory.push(Experience(state, action, next_state, reward))
        return next_state

    def select_action(self, state, evaluate=False):
        self.current_step += 1
        rate = self.strategy.get_exploration_rate(self.current_step) if not evaluate else eps_evaluation
        if rate < random.random():
            # exploit
            with torch.no_grad():
                a = self.policy_network(state)
                b = a.argmax(dim=1)
                c = b.to(self.device)
                return c
        else:
            # explore
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)

    def optimize(self):
        """
             - checks if we have enough memory to take out a batch of our batch_size
             - creates tensors for each of the different values in our experiences
             - Finds the current q-value we have in our policy using the states and actions we have so far
             - Finds the next q-value we have in our policy using the next states we have so far
             - We use the next q-value to calculate what the target q-value is using our discount rate and rewards
        """
        optimizer = optim.Adam(params=self.policy_network.parameters(), lr=learning_rate)
        if self.memory.can_provide_sample(batch_size):
            experiences = self.memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)
            current_q_values = QValues.get_current(self.policy_network, states, actions)
            next_q_values = QValues.get_next(self.target_network, next_states)
            target_q_values = (next_q_values * gamma) + rewards
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
