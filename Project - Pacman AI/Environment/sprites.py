import pygame.sprite

from paths import Paths
from util import *

IMAGE_PATH = PATH + 'sprites/'
# Read tiles from file
PATHS = Paths(readTiles())


class Sprite(pygame.sprite.Sprite):
    """
    Basic Sprite class with most of whats needed in the game, keeps track of position, start position, current image,
    original image and its size
    """

    def __init__(self, position: tuple, image: str, scale: float = 1):
        """
        :param position: (x, y): Start position of this sprite, when resetting sprite will return here
        :param image: filename of default image, when resetting, image will return to this image
        :param scale: Scale the image based on the tile_size
        """
        pygame.sprite.Sprite.__init__(self)
        self.SIZE = (int(TILE_SIZE * scale), int(TILE_SIZE * scale))
        self.image = self.imageDefault = self.loadImage(image)
        self.rect = self.image.get_rect()
        self.position = self.startPosition = position
        self.direction = DEFAULT
        self.offset = (0, 0)  # offset ghosts slightly to see them even if they occupy the same tile

    def reset(self):
        self.position = self.startPosition
        self.direction = DEFAULT
        self.image = self.imageDefault

    def update(self):
        """ Center sprite on current position """
        if self.position == self.startPosition:
            self.rect.center = positionToTileCenter(self.position)
        else:
            self.rect.center = positionToTileCenter(self.position, offset=self.offset)

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def loadImage(self, filename: str):
        return pygame.transform.scale(pygame.image.load(IMAGE_PATH + filename), self.SIZE)


class Target(Sprite):

    def __init__(self, position, ghost):
        super().__init__(position, image=ghost + '-target.png')


class Pellet(Sprite):
    def __init__(self, position: tuple, reward: int = REWARD_PELLET):
        super().__init__(position, 'pellet.png', scale=PELLET_SCALE)
        self.reward = reward
        self.isCollected = False

    def reset(self):
        self.isCollected = False

    def update(self):
        self.kill() if self.isCollected else super().update()

    def collect(self) -> int:
        """ Pick up this specific collectable

        :return: Reward for picking up this collectable
        """
        self.isCollected = True
        return self.reward


class Pacman(Sprite):
    def __init__(self, position):
        super().__init__(position, 'pacman.png', scale=PACMAN_SCALE)
        # Load extra image
        self.imageClosed = self.loadImage('pacman-closed.png')
        self.mouthIsOpen = True
        self.hasMoved = False
        self.direction = RIGHT
        self.update()

    def reset(self):
        self.direction = RIGHT
        super().reset()

    def moveInDirection(self, direction: tuple):
        curNode = PATHS.getNode(self.position)
        # move in direction of action if it is possible
        if direction == UP and curNode.canMoveUp:
            nextNode = PATHS.getNode_Up(curNode)
        elif direction == DOWN and curNode.canMoveDown:
            nextNode = PATHS.getNode_Down(curNode)
        elif direction == LEFT and curNode.canMoveLeft:
            if self.position == self.startPosition:
                nextNode = curNode
            else:
                nextNode = PATHS.getNode_Left(curNode)
        elif direction == RIGHT and curNode.canMoveRight:
            nextNode = PATHS.getNode_Right(curNode)
        else:
            # continue in current direction, regardless of if it is possible or not
            direction = self.direction
            nextNode = PATHS.getNodeInDirection(curNode, direction)
        self.setDirection(direction)
        self.moveTo(nextNode.position)

    def moveTo(self, position: tuple):
        """ Updates position of pacman

        :param position: new coordinate for pacman (x, y)
        """
        self.hasMoved = False if self.position == position else True
        self.position = position

    def setDirection(self, direction: tuple):
        """ Update PacMans visual direction

        :param direction: (x direction, y direction)
        """
        self.direction = direction

    def update(self):
        # Alternate between open and closed image when moving to new tile
        if self.hasMoved:
            self.hasMoved = False
            if self.mouthIsOpen:
                self.image = self.imageClosed
                self.mouthIsOpen = False
            else:
                self.image = self.imageDefault
                self.mouthIsOpen = True
        else:  # reset current image to flip it correctly
            self.image = self.imageDefault if self.mouthIsOpen else self.imageClosed
        # Rotate image in correct direction
        if self.direction == UP:
            self.image = pygame.transform.rotate(self.image, 90)
        elif self.direction == DOWN:
            self.image = pygame.transform.rotate(self.image, -90)
        elif self.direction == LEFT:
            self.image = pygame.transform.flip(self.image, True, False)
        super().update()


class Ghost(Sprite):
    def __init__(self, position: tuple, ghost: str, scatterTarget: tuple, stepsOutOfGhostHouse, offset):
        super().__init__(position, image=ghost + '.png', scale=GHOST_SCALE)
        # Load images
        self.imageUp = self.loadImage(ghost + '-up.png')
        self.imageDown = self.loadImage(ghost + '-down.png')
        self.imageLeft = self.loadImage(ghost + '-left.png')
        self.imageRight = self.loadImage(ghost + '-right.png')
        # Adding variables
        self.scatterTarget = scatterTarget
        self.targetSprite = Target(self.position, ghost)
        self.currentPath, self.currentDist = [], 1000
        self.remainingStepsOutOfGhostHouse = self.stepsOutOfGhostHouse = stepsOutOfGhostHouse
        self.offset = multiply((TILE_SIZE / 8, TILE_SIZE / 8), offset)
        self.pelletsCollectedBeforeSpawning = 0

    def reset(self):
        self.remainingStepsOutOfGhostHouse = self.stepsOutOfGhostHouse
        self.targetSprite.position = self.startPosition
        self.currentPath, self.currentDist = [], 100
        super().reset()

    def update(self):
        """ Update current image based on direction"""
        if self.direction == UP:
            self.image = self.imageUp
        elif self.direction == DOWN:
            self.image = self.imageDown
        elif self.direction == LEFT:
            self.image = self.imageLeft
        elif self.direction == RIGHT:
            self.image = self.imageRight
        else:
            self.image = self.imageDefault
        super().update()

    @staticmethod
    def shouldScatter(currentSecond: float):
        """ Check if the ghosts should scatter based on the amount of pellets pacman has collected

        :param currentSecond: what seconds we are in within the game
        """
        for startSecond, endSecond in SCATTER_AT:
            if startSecond <= currentSecond < endSecond:
                return True
        return False

    def stepTowards(self, target: tuple):
        """ Take a step towards a target position using the pathfinding algorithm

        :param target: (x, y) target position
        """
        self.targetSprite.position = target
        # find path to target
        self.currentPath, self.currentDist, self.direction = PATHS.getPath(self.position, target, self.direction)
        # make sure ghost dont skip a tile if they turn left right after spawning
        if self.position == GHOST_SPAWN and self.currentPath[1].position == (12, 11):
            self.position = (13, 11)
        else:
            self.position = self.currentPath[1].position

    def scatter(self):
        """
        In order for the player not to get trapped by the ghost they routinely scatter to their own corner, in the
        original game this is based on time, but here this is dependant on the score of the player where a higher score
        means they scatter less often.
        """
        self.stepTowards(self.scatterTarget)

    def move(self, amount: float, direction: tuple):
        self.direction = direction
        self.position = add(self.position, multiply(direction, (amount, amount)))

    def takeStepOutOfGhostHouse(self):
        """ Sets target position over exit of ghost house, and reduces remaining steps by one, the classes should
        override this and make sure ghost moves in correct direction to reach target tile """
        self.targetSprite.position = GHOST_SPAWN
        self.remainingStepsOutOfGhostHouse -= 1


class Blinky(Ghost):
    def __init__(self, position):
        super().__init__(position, 'blinky', scatterTarget=(25, -3), stepsOutOfGhostHouse=0, offset=(1, 1))
        self.color = (239, 83, 80)

    def performAction(self, pacman, currentSecond):
        self.scatter() if self.shouldScatter(currentSecond) else self.chase(pacman)

    def chase(self, pacman: Pacman):
        """ Blinky simply finds the shortest path from its current position to where pacman is

        :param pacman: Pacman object with its position
        """
        self.stepTowards(pacman.position)


class Pinky(Ghost):
    def __init__(self, position):
        super().__init__(position, 'pinky', scatterTarget=(3, -3), stepsOutOfGhostHouse=6, offset=(-3, -3))
        self.color = (247, 143, 242)

    def performAction(self, pacman, currentSecond):
        if self.remainingStepsOutOfGhostHouse == 0:
            self.scatter() if self.shouldScatter(currentSecond) else self.chase(pacman)
        else:
            self.takeStepOutOfGhostHouse()

    def chase(self, pacman: Pacman):
        """ Pinky tries to predict where pacman will be by targeting the tile 4 tiles ahead of pacman current direction.
        in the original game however an overflow error causes the target to be both 4 tiles up and 4 tiles to the left
        when pacmans direction is upwards, this is also implemented here, and makes tricking pinky a common tactic.

        :param pacman: Pacman object with position and direction
        """
        # find target
        if pacman.direction == UP:
            target = add(pacman.position, (-4, -4))
        else:
            target = add(pacman.position, multiply(pacman.direction, (4, 4)))
        self.stepTowards(target)

    def takeStepOutOfGhostHouse(self):
        # move half tile upwards each time method is called
        self.move(0.5, UP)
        super().takeStepOutOfGhostHouse()


class Inky(Ghost):
    def __init__(self, position):
        super().__init__(position, 'inky', scatterTarget=(27, 30), stepsOutOfGhostHouse=10, offset=(-1, -1))
        self.color = (128, 222, 234)
        self.spawnAfterNrOfPelletsCollected = 30

    def performAction(self, pacman, blinky, currentSecond, collectedPellets):
        if not collectedPellets < self.spawnAfterNrOfPelletsCollected:
            if self.remainingStepsOutOfGhostHouse == 0:
                self.scatter() if self.shouldScatter(currentSecond) else self.chase(pacman, blinky)
            else:
                self.takeStepOutOfGhostHouse()

    def chase(self, pacman: Pacman, blinky: Blinky):
        """ Inky is the most strategic ghost, and works together with blinky to try to ambush pacman, inky sets a target
        two tiles in front of pacman, and then attempts to mirror blinkys position relative to this tile. Essentially
        trapping pacman in the middle.

        :param pacman: Pacman object with position and direction
        :param blinky: Blinky and its position
        """
        # place target two tiles in front of pacman
        if pacman.direction == UP:
            target = add(pacman.position, (-2, -2))
        else:
            target = add(pacman.position, multiply(pacman.direction, (2, 2)))
        # find blinkys distance from this tile
        blinkyDist = subtract(blinky.position, target)
        # our target is on the other side of blinky
        target = subtract(target, blinkyDist)
        self.stepTowards(target)

    def takeStepOutOfGhostHouse(self):
        self.move(0.5, RIGHT) if self.position[0] != GHOST_SPAWN[0] else self.move(0.5, UP)
        super().takeStepOutOfGhostHouse()


class Clyde(Ghost):
    def __init__(self, position):
        super().__init__(position, 'clyde', scatterTarget=(0, 30), stepsOutOfGhostHouse=10, offset=(3, 3))
        self.color = (255, 193, 7)
        self.spawnAfterNrOfPelletsCollected = 100

    def performAction(self, pacman, currentSecond, collectedPellets):
        if not collectedPellets < self.spawnAfterNrOfPelletsCollected:
            if self.remainingStepsOutOfGhostHouse == 0:
                self.scatter() if self.shouldScatter(currentSecond) else self.chase(pacman)
            else:
                self.takeStepOutOfGhostHouse()

    def chase(self, pacman: Pacman):
        """ Clyde behaves the same a blinky UNLESS he is within 8 tiles of pacman, then he will run towards his scatter
        target regardless of where pacman is until he is out of reach of pacman again and he resumes his hunt

        :param pacman: Pacman object with position
        """
        # find distance to pacman
        distanceToPacman = distance(self.position, pacman.position)
        # chase pacman, but run away if pacman is too close
        self.stepTowards(target=pacman.position) if distanceToPacman >= 8 else self.scatter()

    def takeStepOutOfGhostHouse(self):
        self.move(0.5, LEFT) if self.position[0] != GHOST_SPAWN[0] else self.move(0.5, UP)
        super().takeStepOutOfGhostHouse()
