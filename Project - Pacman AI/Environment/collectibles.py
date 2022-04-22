import pygame

from sprites import Pellet
from globals import PATH


class Collectibles:
    """
    Class that handles all the collectibles available in the game, keeps track of them, resets them after a game is over
    and .....
    """

    def __init__(self):
        self.totalCollected = 0
        self.pellets = []
        self.reset()

    def collect(self, position: tuple) -> int:
        """ Try to collect from a position. This returns the reward if there i a collectible here

        :param position: (x, y) position to attempt to collect from
        :return: Collected reward, return 0 if there were no reward
        """
        for pellet in self.pellets:
            if pellet.position == position:
                if pellet.isCollected:
                    return 0
                else:
                    self.totalCollected += 1
                    return pellet.collect()
        return 0

    def allAreCollected(self) -> bool:
        """ Check if all the pickups have been collected

        :return: True if all are collected, false otherwise
        """
        return self.totalCollected == len(self.pellets)

    def reset(self):
        self.totalCollected = 0
        self.pellets = []
        with open(PATH + "data/collectibles.txt", 'r') as file:
            for y, line in enumerate(file):
                for x, char in enumerate(line):
                    if char == "1":
                        collectable = Pellet((x, y))
                        self.pellets.append(collectable)

    def update(self):
        for collectable in self.pellets:
            collectable.update()

    def draw(self, screen):
        for collectable in self.pellets:
            if not collectable.isCollected:
                collectable.draw(screen)
