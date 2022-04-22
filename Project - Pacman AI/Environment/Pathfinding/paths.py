from globals import UP, DOWN, LEFT, RIGHT
from node import Node
from util import add, multiply, getDirection


class Paths:

    def __init__(self, walls: list):
        self.nodes = self.__createNodes(walls)

    def getPath(self, start: tuple, end: tuple, currentDirection: tuple = (0, 0)) -> (list, int, tuple):
        """ Finds the shortest path from start position to end position

        :param start: Current node position
        :param end: Goal node position
        :param currentDirection: Ghosts cannot move directly backwards
        :return: - List of nodes to go through <br/>
                 - Total distance from start to end
                 - Direction to first tile on path
        """

        # Create start and end node
        startNode = self.__findClosestNode(start)
        endNode = self.__findClosestNode(end)
        if startNode == endNode:
            # move target to previous tile
            endNode = self.getNodeInDirection(endNode, multiply((-1, -1), currentDirection))

        # Initialize both open and closed list, open are nodes to check, closed are already checked
        open_list = []
        closed_list = []

        # Add the start node
        open_list.append(startNode)

        # Manually add connected nodes (children), excluding illegal directions, for starting node
        children = self.__getChildren(startNode, currentDirection)

        # Loop until you find the end, if list is empty there is no path
        while len(open_list) > 0:

            # Check a new node
            currentNode = self.__getNextNode(open_list, closed_list)

            # Found the goal
            if currentNode == endNode:
                distance = currentNode.distanceFromStart
                path = []
                current = currentNode
                while current is not None:
                    path.append(current)
                    current = current.parent

                if len(path) < 2:
                    path.append(path[0])

                # reverse path
                path = path[::-1]
                # find direction of path from start position
                direction = getDirection(start, path[1].position) if len(path) > 2 else currentDirection

                # Reset nodes before next search
                for node in closed_list:
                    node.reset()
                for node in open_list:
                    node.reset()

                # Return shortest path, distance and direction from current position
                return path, distance, direction

            if currentNode != startNode:
                # Add Children for this node
                children = self.__getChildren(currentNode)

            # Loop through children and check if we want to add it to the open_list
            for child in children:

                # Child has already been checked
                if child in closed_list:
                    continue

                if child in open_list:
                    # Child is already in open list
                    for index, open_node in enumerate(open_list):
                        # Node is already in the open list
                        if child == open_node:
                            # update node in list if new path is shorter
                            if currentNode.distanceFromStart < open_node.distanceFromStart - 1:
                                self.__setValues(node=open_list[index], parent=currentNode, endNode=endNode)
                            break
                else:
                    self.__setValues(node=child, parent=currentNode, endNode=endNode)

                    # Add the child to the open list
                    open_list.append(child)

        print("NO PATH FOUND FOR: start:", start, ":", startNode, "end:", end, ":", endNode)

    def getNode(self, position: tuple) -> Node:
        """ Returns the node in a given position

        :param position: (x, y) position of the node
        :return: Node in the given position, returns None if there isn't a node here
        """
        x, y = int(position[0]), int(position[1])
        # Make sure coordinates are within map
        if x < 0 or y < 0:
            return None
        if x >= len(self.nodes[0]) or y >= len(self.nodes):
            return None
        # return node in given position, is None if it isn't a path node
        return self.nodes[y][x]

    def getNodeInDirection(self, start: Node, direction: tuple):
        if direction == UP:
            return self.getNode_Up(start)
        elif direction == DOWN:
            return self.getNode_Down(start)
        elif direction == LEFT:
            return self.getNode_Left(start)
        elif direction == RIGHT:
            return self.getNode_Right(start)
        else:
            return start

    def getNode_Up(self, currentNode: Node) -> Node:
        """ Returns node if you move up from the current node

        :param currentNode: current node position
        :return: node after moving up
        """
        if not currentNode.canMoveUp:
            return self.getNode(currentNode.position)
        return self.getNode(add(currentNode.position, UP))

    def getNode_Down(self, currentNode: Node) -> Node:
        """ Returns node if you move down from the current node

        :param currentNode: current node position
        :return: node after moving down
        """
        if not currentNode.canMoveDown:
            return self.getNode(currentNode.position)
        return self.getNode(add(currentNode.position, DOWN))

    def getNode_Left(self, currentNode: Node) -> Node:
        """ Returns node if you move left from the current node

        :param currentNode: current node position
        :return: node after moving to the left
        """
        if not currentNode.canMoveLeft:
            return self.getNode(currentNode.position)
        if currentNode.position == (1, 14):
            return self.getNode((26, 14))
        return self.getNode(add(currentNode.position, LEFT))

    def getNode_Right(self, currentNode: Node) -> Node:
        """ Returns node if you move right from the current node

        :param currentNode: current node position
        :return: node after moving to the right
        """
        if not currentNode.canMoveRight:
            return self.getNode(currentNode.position)
        if currentNode.position == (26, 14):
            return self.getNode((1, 14))
        return self.getNode(add(currentNode.position, RIGHT))

    @staticmethod
    def __getNextNode(open_list: list, closed_list: list) -> Node:
        """ Get the next node, this is the node with the currently shortest estimated distance

        :param open_list: List off all open nodes, ie. nodes we have found but not checked yet
        :param closed_list: List of all nodes we have already checked
        :return:
        """
        node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            # Find node with lowest estimated distance to the end
            if item.estimatedTotalDistanceToEnd < node.estimatedTotalDistanceToEnd:
                node = item
                current_index = index
        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(node)
        return node

    @staticmethod
    def __setValues(node: Node, parent: Node, endNode: Node) -> None:
        """

        :param node:
        :param parent:
        :return:
        """
        # set parent node
        node.parent = parent
        # Create distance values for the node
        node.distanceFromStart = parent.distanceFromStart + 1
        node.estimatedDistanceToEnd = node.distanceToNode(endNode)
        node.estimatedTotalDistanceToEnd = node.distanceFromStart + node.estimatedDistanceToEnd

    def __getChildren(self, node: Node, currentDirection: tuple = (0, 0)) -> list:
        """ Finds all the nodes connected to the current node

        :param node: Node to get connections from
        :param currentDirection: Block a path in a certain direction
        :return: list of nodes (children) this node is connected to
        """
        children = []
        if currentDirection != DOWN:
            children.append(self.getNode_Up(node))
        if currentDirection != UP:
            children.append(self.getNode_Down(node))
        if currentDirection != RIGHT:
            children.append(self.getNode_Left(node))
        if currentDirection != LEFT:
            children.append(self.getNode_Right(node))
        return [child for child in children if child is not node]

    def __findClosestNode(self, position: tuple) -> Node:
        """ Find the closes node to the position specified

        :param position: (x, y)
        :return: The closest node to the end Position
        """
        x, y = position

        # make sure coordinates are within the grid
        x = max(x, 1)
        x = min(x, 26)
        y = max(y, 1)
        y = min(y, 29)

        # Check if end node exists
        actual = self.getNode((x, y))
        if actual is not None:
            return actual

        # Find closest node to end node
        node = None
        i = 1
        while node is None:
            # check upwards
            node = self.getNode((x, y - i))
            if node is not None:
                break
            # check downwards
            node = self.getNode((x, y + i))
            if node is not None:
                break
            # check to the left
            node = self.getNode((x - i, y))
            if node is not None:
                break
            # check to the right
            node = self.getNode((x + i, y))
            # increase distance from goal position
            i = i + 1
        return node

    @staticmethod
    def __createNodes(tiles: list) -> list:
        """ Reads all nodes from the file, the nodes contain info about all the tiles pacman and the ghost can move
        in, this includes coordinates, valid moves in that position, distance to ghost spawn etc.

        :return: List of nodes from file
        """
        nodes = []
        PATH, WALL, GATE, EMPTY = "0", "1", "2", "3"

        # figure out how to fill each tile based on connected tiles
        maxX, maxY = len(tiles[0]), len(tiles)
        for y in range(maxY):
            row = []
            for x in range(maxX):
                if tiles[y][x] == PATH and x != 0 and x != 27:
                    node = Node(position=(x, y))
                    if tiles[y][(x - 1) % maxX] == PATH:
                        node.canMoveLeft = True
                    if tiles[y][(x + 1) % maxX] == PATH:
                        node.canMoveRight = True
                    if tiles[(y - 1) % maxY][x] == PATH:
                        node.canMoveUp = True
                    if tiles[(y + 1) % maxY][x] == PATH:
                        node.canMoveDown = True
                    row.append(node)
                else:
                    row.append(None)
            nodes.append(row)
        return nodes
