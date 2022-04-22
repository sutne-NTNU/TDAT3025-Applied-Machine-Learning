import os

PATH = os.path.join(os.path.dirname(__file__), '')

# Height and with of each tile on the screen
TILE_SIZE = 30

# Scale of sprites
GHOST_SCALE = 1.7
PACMAN_SCALE = 1.7
PELLET_SCALE = 1.1

# Rewards
REWARD_PELLET = 10
REWARD_KILLED = -100
REWARD_WON = 1000

# Number of steps per second to emulate the timer in the real game
STEPS_PER_SECOND = 7
# Start to scatter after the following amount of "seconds" in format (start second, end second)
SCATTER_AT = [(0, 7), (27, 34), (54, 59), (79, 84)]

# Position of ghosts after they are done moving out of the ghost house
GHOST_SPAWN = (13.5, 11)

# Directional tuples
DEFAULT = (0, 0)
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
