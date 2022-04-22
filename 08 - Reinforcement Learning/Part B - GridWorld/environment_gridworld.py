import math
import pygame
import sys
import random
import numpy as np
import time


class Environment:

    def __init__(self, gridsize, obsticlePercentage=0.10, bonusPercentage=0.01, dangerPercentage=0.01):
        # Create window and grid
        self.gridsize = gridsize
        self.WINDOW_HEIGHT = 900
        self.WINDOW_WIDTH = 900
        self.SCREEN = pygame.display.set_mode((self.WINDOW_HEIGHT, self.WINDOW_WIDTH))
        self.TILE_SIZE = self.WINDOW_HEIGHT // self.gridsize

        # Place goal in center of grid
        self.goal = math.floor(gridsize / 2), math.floor(gridsize / 2)
        self.state = (0, 0)

        # Create obstacles, bonuses (positive reward) and dangers (negative reward)
        self.obstacles = self.__createObstacles(obsticlePercentage)
        self.dangers = self.__createDangers(dangerPercentage)
        self.bonuses = Bonuses(self.gridsize, self.goal, self.obstacles, self.dangers, bonusPercentage=bonusPercentage)

        # All possible actions in this environment
        self.actions = ["up", "down", "left", "right"]

        # Setting colours of the different objects in the environment
        self.BACKGROUND = '#162330'
        self.GOAL = '#fff700'
        self.AGENT = '#0095ff'
        self.BONUS = '#fffca6'
        self.DANGER = '#ff91b6'
        self.OBSTACLE = '#000000'

    def reset(self):
        """set agent state (location) to random location in the grid"""
        self.state = (random.randint(0, self.gridsize - 1), random.randint(0, self.gridsize - 1))
        if self.state in self.obstacles:
            return self.reset()
        # reset bonuses that have been picked up
        self.bonuses.reset()
        return self.state

    def step(self, action_index):
        """Take a step"""
        reward, done = 0, False
        # get current state
        x, y = self.state[0], self.state[1]
        # change state based on action
        action = self.actions[action_index]
        if action == "up":
            y -= 1
        elif action == "down":
            y += 1
        elif action == "left":
            x -= 1
        elif action == "right":
            x += 1

        # Check if next state is goal
        if (x, y) == self.goal:
            reward = 100
            done = True
        # check if next state is outside of grid
        if x >= self.gridsize or y >= self.gridsize or x < 0 or y < 0:
            (x, y) = self.state
        # check if next state is an obstacle
        if (x, y) in self.obstacles:
            (x, y) = self.state
        # check if next state is a danger
        if (x, y) in self.dangers:
            reward = -25
        # check if agent found a bonus reward
        if self.bonuses.isBonus((x, y)):
            reward = self.bonuses.getBonus((x, y))

        self.state = (x, y)
        return self.state, reward, done

    def render(self, q_table=None, speed=100):
        """Render the environment, if a q_table is provided will visalize the optimal action in each location"""
        self.SCREEN.fill(self.BACKGROUND)

        self.__draw(q_table)

        # lower value means faster movement
        time.sleep(1 / speed)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()

    def __draw(self, q_table=None):
        for x in range(self.gridsize):
            for y in range(self.gridsize):
                # Draw obstacles
                if (x, y) in self.obstacles:
                    self.__fillTile((x, y), color=self.OBSTACLE)
                    continue
                # Draw dangers
                if (x, y) in self.dangers:
                    self.__fillTile((x, y), color=self.DANGER)
                    continue
                # Draw bonus
                if self.bonuses.isBonus((x, y)) > 0:
                    self.__fillTile((x, y), color=self.BONUS)
                    continue

                # Add background
                self.__fillTile((x, y), color=self.BACKGROUND)
                # visualize q-table values in position (x,y) if they arent all zero
                if q_table is not None:
                    if not np.max([abs(n) for n in q_table[x][y]]) == 0:
                        self.__fillTile((x, y), color=self.__gradient(q_table, (x, y)))

        # draw player and goal
        self.__fillTile((self.state[0], self.state[1]), color=self.AGENT)
        self.__fillTile((self.goal[0], self.goal[1]), color=self.GOAL)

    def __fillTile(self, position, color=None, image=None):
        """Takes x and y coordinate of square and fills it with given color or image"""
        # create square to fill
        sqr = pygame.Rect(position[0] * self.TILE_SIZE,
                          position[1] * self.TILE_SIZE,
                          self.TILE_SIZE, self.TILE_SIZE)
        if image is not None:
            # fill sqr with image
            self.SCREEN.blit(image, sqr)
        if color is not None:
            # fill sqr with color
            pygame.draw.rect(self.SCREEN, color, sqr)

    @staticmethod
    def __gradient(q_table, poition):
        """Returns the colour of the position relative to all the other values in the q_table, green=high, red=low"""
        value = np.max(q_table[poition[0]][poition[1]])
        min_value, max_value = np.min(q_table), np.max(q_table)
        percentage = value / (max_value - min_value)
        green = math.floor(110 * percentage)
        red = math.floor(100 - 100 * percentage)
        return red, green, 0

    def __createObstacles(self, obstaclePrecentage):
        obstacles = []
        # randomly place obstacles
        for x in range(self.gridsize):
            for y in range(self.gridsize):
                if random.uniform(0, 1) < obstaclePrecentage and (x, y) != self.goal:
                    obstacles.append((x, y))
        # if a square has 3 connected obstacles and/or border squares, this square should also be a obstacle
        for x in range(self.gridsize):
            for y in range(self.gridsize):
                connected = 0
                if (x + 1, y) in obstacles or x + 1 < 0 or x + 1 > self.gridsize:
                    connected += 1
                if (x - 1, y) in obstacles or x - 1 < 0 or x - 1 > self.gridsize:
                    connected += 1
                if (x, y + 1) in obstacles or y + 1 < 0 or y + 1 > self.gridsize:
                    connected += 1
                if (x, y - 1) in obstacles or y - 1 < 0 or y - 1 > self.gridsize:
                    connected += 1
                if connected >= 3:
                    obstacles.append((x, y))
        return obstacles

    def __createDangers(self, dangerPercentage):
        dangers = []
        # randomly place obstacles
        for x in range(self.gridsize):
            for y in range(self.gridsize):
                if random.uniform(0, 1) < dangerPercentage and (x, y) != self.goal and (x, y) not in self.obstacles:
                    dangers.append((x, y))
        return dangers


class Bonuses:
    """A bonus gives a small reward when picked up

    The bonus is only picked up the first time an agent steps into that tile, resets every run"""

    def __init__(self, gridsize, goal, obstacles, dangers, bonusPercentage):
        self.gridsize = gridsize
        self.bonus = 10
        self.bonuses = self.__createBonuses(bonusPercentage, goal, obstacles, dangers)

    def __createBonuses(self, bonusPercentage, goal, obstacles, dangers):
        bonuses = {}
        # randomly place obstacles
        for x in range(self.gridsize):
            for y in range(self.gridsize):
                if random.uniform(0, 1) < bonusPercentage \
                        and (x, y) != goal \
                        and (x, y) not in obstacles \
                        and (x, y) not in dangers:
                    bonuses[(x, y)] = self.bonus
        return bonuses

    def reset(self):
        for key in self.bonuses.keys():
            self.bonuses[key] = self.bonus

    def isBonus(self, key):
        if key in self.bonuses:
            return self.bonuses[key] > 0
        return False

    def getBonus(self, key):
        bonus = self.bonuses[key]
        self.bonuses[key] = 0
        return bonus
