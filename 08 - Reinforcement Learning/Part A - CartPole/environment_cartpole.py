from sklearn.preprocessing import KBinsDiscretizer
import math
import gym


class Environment:

    def __init__(self, environment, n_bins=(6, 12)):
        # Load cartpole environment from "gym"
        self.env = gym.make(environment)
        self.n_bins = n_bins
        # split continous observation of angle of the pole into bins, starting at lowest, until highest
        self.lower_bounds = [self.env.observation_space.low[2], -math.radians(50)]
        self.upper_bounds = [self.env.observation_space.high[2], math.radians(50)]

        self.action_space = self.env.action_space

    def discretizer(self, _, __, angle, pole_velocity):
        """Convert continous state into a discrete state"""
        est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        est.fit([self.lower_bounds, self.upper_bounds])
        return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))

    def step(self, action):
        """perform the given action and return new state, reward and status"""
        obs, reward, done, info = self.env.step(action)
        next_state = self.discretizer(*obs)
        return next_state, reward, done

    def reset(self):
        """resets environment, and return starting state to the agent"""
        return self.discretizer(*self.env.reset())

    def render(self):
        self.env.render()

    @staticmethod
    def solved(scores):
        """Solved if the average score is above 195 for the last 100 episodes"""
        if len(scores) < 100:
            return False

        counter, total = 0, 0
        for ep in reversed(scores):
            counter += 1
            total += ep
            if counter >= 100:
                break
        return total / 100 > 195
