import random


class ReplayMemory:
    """
    Used for training our DQN, by taking batches of our replay memory and reusing it later (?)
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []  # where we store our experiences
        self.push_count = 0  # Keeps track of how many experiences we have added to memory

    def push(self, experience):
        """
        method for adding experiences form our training to our replay memory
        :param experience: the observation from that state, what action it took and what reward it got for doing so
                            and what state was the next
        """
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        """
        :param batch_size: thee number of experiences we want to sample
        :return: random sample of experiences from memory
        """
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        """
        :param batch_size: nr of experiences we want from memory
        :return: True if we have enough memory to sample a batch matching the batch size provided False if we can't
        """
        return len(self.memory) >= batch_size
