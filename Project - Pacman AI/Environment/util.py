import math
import matplotlib.pyplot as plt

from globals import *


def readTiles() -> list:
    """ Read the tiles from file, this means walls, empty space, paths etc.

    :return: Two dimensional list with numbered tiles
    """
    tiles = []
    with open(PATH + "data/tiles.txt", 'r') as file:
        for line in file:
            row = []
            for char in line:
                if char != "\n":
                    row.append(char)
            tiles.append(row)
    return tiles


def positionToTileCenter(position: tuple, offset: tuple = (0, 0)) -> tuple:
    """ Converts coordinates to find center-pixel of tile they are located in

    :param position: (x, y) coordinates within the game
    :param offset: offset in pixels in each direction
    :return:
    """
    x, y = position
    # create offset from top
    verticalOffset = 3 * TILE_SIZE
    # find top left pixel of tile
    topLeftX = x * TILE_SIZE
    topLeftY = y * TILE_SIZE + verticalOffset
    # Shift start relative to tile size to center area within a tile
    centerX = topLeftX + (TILE_SIZE // 2)
    centerY = topLeftY + (TILE_SIZE // 2)
    center = add((centerX, centerY), offset)
    return center


def positionToScreenPosition(position: tuple, scale: float = 1.0):
    """ Converts coordinates to screen coordinates, ie. top left pixel

    :param position:
    :param scale:
    :return:
    """
    # Get center of tile
    center = positionToTileCenter(position)
    x, y = center
    # find top left pixel based on scale
    topLeftX = x - (TILE_SIZE * scale // 2)
    topLeftY = y - (TILE_SIZE * scale // 2)
    topLeftPixel = (topLeftX, topLeftY)
    return topLeftPixel


def multiply(tuple1: tuple, tuple2: tuple) -> tuple:
    """ Multiplies two tuples together and returns result

    :param tuple1:
    :param tuple2:
    :return: tuple with result
    """
    x1, y1 = tuple1
    x2, y2 = tuple2
    return x1 * x2, y1 * y2


def add(tuple1: tuple, tuple2: tuple) -> tuple:
    """ Adds two tuples together and returns result

    :param tuple1:
    :param tuple2:
    :return: tuple with result
    """
    x1, y1 = tuple1
    x2, y2 = tuple2
    return x1 + x2, y1 + y2


def subtract(tuple1: tuple, tuple2: tuple) -> tuple:
    """ subtract tuple2 from tuple1

    :param tuple1:
    :param tuple2:
    :return: tuple with result
    """
    x1, y1 = tuple1
    x2, y2 = tuple2
    return x1 - x2, y1 - y2


def getDirection(start: tuple, end: tuple) -> tuple:
    """ Find direction between two positions

    :param start:
    :param end:
    :return: tuple with direction (x-direction, y-direction)
    """
    x1, y1 = start
    x2, y2 = end
    dirX, dirY = x2 - x1, y2 - y1
    # check teleport directions
    if dirX > 3:
        return LEFT
    elif dirX < -3:
        return RIGHT
    # make directions either -1 or 1
    if dirX != 0:
        dirX = (dirX / abs(dirX))
    if dirY != 0:
        dirY = (dirY / abs(dirY))
    return dirX, dirY


def distance(tuple1: tuple, tuple2: tuple) -> float:
    """ Find distance between two tuples of coordinates

    :param tuple1:
    :param tuple2:
    :return: tuple with result
    """
    x1, y1 = tuple1
    x2, y2 = tuple2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def stepsToSeconds(stepCount: int) -> float:
    """ Convert number of steps to number of seconds """
    return stepCount / STEPS_PER_SECOND


score_file = 'data/human_scores.txt'


def plot_human_scores():
    scores = []
    file = open(PATH + score_file)
    for line in file:
        scores.append(int(line))
    plt.plot(scores, '.', color='0.65', label="Score")
    average = sum(scores) / len(scores)
    plt.plot([0, len(scores) - 1], [average, average], '-', color='r', label="Gjennomsnittlig Score: %i" % average)
    plt.legend()
    plt.ylim(0, 2440)
    plt.title('Menneskelig Spiller')
    plt.xlabel('Antall spill')
    plt.savefig('./media/human_scores.png')
    plt.show()


def save_human_score(score):
    file = open(PATH + score_file, 'a+')
    file.write("%i\n" % score)
