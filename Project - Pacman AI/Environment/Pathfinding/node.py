from util import distance


class Node:

    def __init__(self, position: tuple):
        # Location
        self.position = position
        # Attributes
        self.canMoveUp = False
        self.canMoveDown = False
        self.canMoveLeft = False
        self.canMoveRight = False
        # Pathfinding
        self.parent = None
        self.distanceFromStart = 0
        self.estimatedDistanceToEnd = 0
        self.estimatedTotalDistanceToEnd = 0

    def distanceToNode(self, other):
        if self is None or other is None:
            return 1
        return distance(self.position, other.position)

    def reset(self):
        self.parent = None
        self.distanceFromStart = 0
        self.estimatedDistanceToEnd = 0
        self.estimatedTotalDistanceToEnd = 0

    def __eq__(self, other):
        if self is None or other is None:
            return False
        return self.position == other.position

    def __str__(self):
        x, y = self.position
        return "(" + str(x) + " , " + str(y) + ")"
