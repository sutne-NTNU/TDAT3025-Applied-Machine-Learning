import sys
import time

import pygame

from collectibles import Collectibles
from paths import Paths
from sprites import Pacman, Blinky, Inky, Pinky, Clyde
from util import *


class Environment:

    def __init__(self):
        """Initialize all environment variables"""
        # Set program icon and window name
        pygame.display.set_icon(pygame.image.load(PATH + 'sprites/blinky-right.png'))
        pygame.display.set_caption('Pacman')
        # Initialize window
        self.WINDOW_WIDTH = TILE_SIZE * 28  # game is 28 tiles wide
        self.WINDOW_HEIGHT = TILE_SIZE * (31 + 3)  # 31 tiles for game, 3 for score (on top)
        self.SCREEN = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))

        # Text
        pygame.init()
        self.FONT = pygame.font.SysFont('Calibri', TILE_SIZE * 2)

        # Movable path tiles with information about shortest path, nodes etc.d
        self.map = readTiles()
        self.paths = Paths(self.map)

        # All collectibles
        self.collectibles = Collectibles()

        # All possible actions in this environment
        self.actions = ["up", "down", "left", "right"]
        self.score = self.currentStep = 0

        # Setting colours of the different objects in the environment
        self.WALL_COLOR = (30, 28, 26)
        self.BACKGROUND_COLOR = (16, 16, 16)

        # Creating PACMAN
        self.pacman = Pacman(position=(13.5, 23))

        # Creating the ghosts
        self.blinky = Blinky(position=(13.5, 11))
        self.inky = Inky(position=(11.5, 14))
        self.pinky = Pinky(position=(13.5, 14))
        self.clyde = Clyde(position=(15.5, 14))
        self.ghosts = pygame.sprite.Group(self.blinky, self.pinky, self.inky, self.clyde)

    def reset(self):
        """ Revert environment back to its original state, this includes ghost positions, pac-man position resetting
        pickups and score etc.

        :return: (x, y): initial position of pacman
        """
        self.score = self.currentStep = 0
        self.collectibles.reset()
        self.pacman.reset()
        for ghost in self.ghosts:
            ghost.reset()
        return self.pacman.position

    def step(self, action_index: int) -> (tuple, int, bool):
        """ Perform an action from the current state of pacman, calculate movements and potential rewards and return
        rewards and states

        :param action_index: index of action pacman wants to make
        :return: - tuple (x, y): next position of pacman <br/>
                 - Reward for taking this step <br/>
                 - If the game is over or not
        """
        self.currentStep += 1
        currentSecond = stepsToSeconds(self.currentStep)

        reward, done = 0, False
        # change position based on action
        direction = DEFAULT
        if action_index is not None:
            action = self.actions[action_index]
            if action == "up":
                direction = UP
            elif action == "down":
                direction = DOWN
            elif action == "left":
                direction = LEFT
            elif action == "right":
                direction = RIGHT

        # Move pacman
        self.pacman.moveInDirection(direction)

        # get reward if there is one in next position
        reward += self.collectibles.collect(self.pacman.position)
        self.score += reward

        # Check if all dots are collected
        if self.collectibles.allAreCollected():
            reward += REWARD_WON
            done = True
            return reward, done

        # Check if he moved into a ghost
        if self.pacmanIsKilled():
            reward += REWARD_KILLED
            done = True
            return reward, done

        # Make the ghost perform their action based on their own logic
        self.blinky.performAction(self.pacman, currentSecond)
        self.pinky.performAction(self.pacman, currentSecond)
        self.inky.performAction(self.pacman, self.blinky, currentSecond, self.collectibles.totalCollected)
        self.clyde.performAction(self.pacman, currentSecond, self.collectibles.totalCollected)

        # Check if any of the ghosts moved into pacman
        if self.pacmanIsKilled():
            reward += REWARD_KILLED
            done = True

        return reward, done

    def render(self, mode: str = "play", drawTargets=False, drawPaths=False):
        """ Renders the environment and adds a listener for if the window gets closed, also sets the minimum timeout
        between each render.

        :param mode: "play" or "training", in training mode the additional arguments are ignored
        :param timeout: Time between each frame in milliseconds, in training this is set to 0 tos peed up learning
        :param drawTargets: choose to draw the current target tiles of the ghosts
        :param drawPaths: Choose to show the current path the ghost will take towards their target
        :return: The screen that is rendered
        """
        if mode == "training":
            timeout = 0
            drawPaths = drawTargets = False
        else:
            timeout = 66 / 1000
            # timeout = 1 / STEPS_PER_SECOND

        # Draw static map and background
        self.__drawMap(self.map)
        # display current score in text on top of screen
        self.__drawScore()
        # Draw the paths of the ghosts
        self.__drawPaths() if drawPaths else None
        # Draw all collectibles
        self.__drawCollectibles()
        # Draw all movable objects (pacman and ghosts)
        self.__drawEntities()
        # Draw the ghosts targets
        self.__drawTargets() if drawTargets else None

        if mode == "training":
            compressedScreenArray = self.__scaleScreenAndConvertToArray()

        pygame.display.update()

        # timeout between next frame can start to render
        time.sleep(timeout)

        return compressedScreenArray if mode == "training" else None

    def pacmanIsKilled(self) -> bool:
        """ Checks if any of the ghosts positions are equal to pacman both before and after he moves

        :return: True if the ghosts have killed pacman, false otherwise
        """
        for ghost in self.ghosts:
            if self.pacman.position == ghost.position:
                return True
        return False

    def __drawMap(self, tiles: list) -> None:
        """ Draw map using tiles.txt file where the zeroes are the area the pacman can move and ones are the walls """
        # Fill entire environment with background colour
        self.SCREEN.fill(self.BACKGROUND_COLOR)

        path, wall, gate, empty = "0", "1", "2", "3"
        # figure out how to fill each tile based on connected tiles
        maxX, maxY = len(tiles[0]), len(tiles)
        for y in range(maxY):
            for x in range(maxX):
                # keep track off what corners to draw based on connected paths
                tLeft = tRight = bLeft = bRight = True
                if tiles[y][x] == wall:
                    if tiles[y][(x - 1) % maxX] == path:
                        tLeft = bLeft = False
                    if tiles[y][(x + 1) % maxX] == path:
                        tRight = bRight = False
                    if tiles[(y - 1) % maxY][x] == path:
                        tLeft = tRight = False
                    if tiles[(y + 1) % maxY][x] == path:
                        bLeft = bRight = False
                    # check diagonal up left
                    if tiles[(y - 1) % maxY][(x - 1) % maxX] == path:
                        tLeft = False
                    if tiles[(y - 1) % maxY][(x + 1) % maxX] == path:
                        tRight = False
                    if tiles[(y + 1) % maxY][(x - 1) % maxX] == path:
                        bLeft = False
                    if tiles[(y + 1) % maxY][(x + 1) % maxX] == path:
                        bRight = False

                    self.__fillCorner((x, y), self.WALL_COLOR, tLeft, tRight, bLeft, bRight)

                if tiles[y][x] == gate:
                    # special drawing for the gate tiles
                    width, height = TILE_SIZE, TILE_SIZE // 4
                    topLeftX, topLeftY = positionToScreenPosition((x, y))
                    topLeftY = topLeftY + TILE_SIZE // 2
                    corner = pygame.Rect(topLeftX, topLeftY, width, height)
                    pygame.draw.rect(self.SCREEN, self.WALL_COLOR, corner)

    def __fillCorner(self, position: tuple, color,
                     tLeft: bool = True, tRight: bool = True, bLeft: bool = True, bRight: bool = True) -> None:
        """ Fills a tile with either a colour, or scales an image to fit within the tile in the given position

        :param position: (x, y) position of tile to fill
        :param color: String of colour in hexadecimal format or an rgb tuple (R, G, B)
        """
        width = height = TILE_SIZE // 2
        topLeftX, topLeftY = positionToScreenPosition(position)
        if tLeft:
            corner = pygame.Rect(topLeftX, topLeftY, width, height)
            pygame.draw.rect(self.SCREEN, color, corner)
        if tRight:
            corner = pygame.Rect(topLeftX + width, topLeftY, width, height)
            pygame.draw.rect(self.SCREEN, color, corner)
        if bLeft:
            corner = pygame.Rect(topLeftX, topLeftY + height, width, height)
            pygame.draw.rect(self.SCREEN, color, corner)
        if bRight:
            corner = pygame.Rect(topLeftX + width, topLeftY + height, width, height)
            pygame.draw.rect(self.SCREEN, color, corner)

    def __drawScore(self) -> None:
        """
        Displays the score on top of the screen
        """
        textSurface = self.FONT.render("Score: " + str(self.score), True, (255, 255, 255))
        xPos = self.WINDOW_WIDTH // 2 - TILE_SIZE * 3
        yPos = TILE_SIZE * 0.25
        self.SCREEN.blit(textSurface, (xPos, yPos))

    def __drawPaths(self) -> None:
        """ Draw entire path of the ghosts """
        for ghost in self.ghosts.sprites():
            for i in range(1, len(ghost.currentPath) - 1):
                self.__drawPathBetween(ghost.currentPath[i].position,
                                       ghost.currentPath[i + 1].position,
                                       ghost.color,
                                       ghost.offset)

    def __drawPathBetween(self, start: tuple, end: tuple, color, offset=(0, 0)) -> None:
        """ Draw path from one node/position to the next

        :param start: (x, y) coordinate of start tile
        :param end: (x,y) position of end tile
        :param color:
        """
        # Create rectangle of height and width
        offset = multiply((0.5, 0.5), offset)
        width = TILE_SIZE / 6
        offsetFromCenter = width / 2
        height = TILE_SIZE + offsetFromCenter * 2
        # flip rectangle in right direction
        direction = getDirection(start, end)
        if direction == RIGHT or direction == LEFT:
            height, width = width, height
        if direction == UP or direction == LEFT:
            start, end = end, start
        topLeftX, topLeftY = subtract(positionToTileCenter(start, offset), (offsetFromCenter, offsetFromCenter))
        path = pygame.Rect(topLeftX, topLeftY, width, height)
        # draw rectangle/path
        pygame.draw.rect(self.SCREEN, color, path)

    def __drawCollectibles(self) -> None:
        """ Draws all pickups that haven't been picked up yet """
        self.collectibles.update()
        self.collectibles.draw(self.SCREEN)

    def __drawEntities(self) -> None:
        """ Updates and draws all sprites position on the screen"""
        self.pacman.update()
        self.pacman.draw(self.SCREEN)
        self.ghosts.update()
        self.ghosts.draw(self.SCREEN)

    def __drawTargets(self):
        for ghost in self.ghosts:
            ghost.targetSprite.update()
            ghost.targetSprite.draw(self.SCREEN)

    def __scaleScreenAndConvertToArray(self):
        # Scale screen
        scaled = pygame.transform.scale(self.SCREEN, (int(28 * 3.5), int(34 * 3.5)))
        width, height = scaled.get_size()

        # Convert pixels to grayscale
        arr = pygame.surfarray.array3d(scaled)
        arr = arr.dot([0.298, 0.587, 0.114])[:, :, None].repeat(3, axis=2)
        converted = pygame.surfarray.make_surface(arr)

        # blit to screen
        x, y = 35, self.WINDOW_HEIGHT // 2 - (height + TILE_SIZE // 2)
        self.SCREEN.blit(converted, (x, y), pygame.Rect(0, 0, width, height))

        convertedArray = pygame.surfarray.pixels_red(converted)
        return convertedArray

    def play(self, mode="play", hacks="off"):
        """ Manually play in the environment with the arrow keys, you can toggle visual targets and paths for the ghost
        by pressing "T"(targets) and "P"(paths) or "H" to toggle both at the same time, and u can use "Esc" to quit
        """
        up, down, left, right = 0, 1, 2, 3
        drawTargets = drawPaths = paused = False
        if hacks == "on":
            drawTargets = drawPaths = True
        stop = False
        while not stop:
            self.render(mode=mode, drawPaths=drawPaths, drawTargets=drawTargets)
            time.sleep(1)
            done = False
            while not done:
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        stop = done = True
                        break
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            stop = done = True
                            break
                        if event.key == pygame.K_t:
                            drawTargets = not drawTargets
                        if event.key == pygame.K_p:
                            drawPaths = not drawPaths
                        if event.key == pygame.K_h:
                            drawPaths = not drawPaths
                            drawTargets = not drawTargets
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                        self.render(mode=mode, drawPaths=drawPaths, drawTargets=drawTargets)

                if not paused and not done:
                    action = None
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_UP]:
                        action = up
                    elif keys[pygame.K_DOWN]:
                        action = down
                    elif keys[pygame.K_LEFT]:
                        action = left
                    elif keys[pygame.K_RIGHT]:
                        action = right
                    if action is not None:
                        reward, done = self.step(action)
                    self.render(mode=mode, drawPaths=drawPaths, drawTargets=drawTargets)
                    if done:
                        print("You got a score of " + str(self.score))
                        save_human_score(self.score)
                        time.sleep(1)
                        self.reset()

        plot_human_scores()
