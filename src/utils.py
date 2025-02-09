import numpy as np
import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_dataclass_from_dict(dataclass, data_dict, convert_list_to_array=False):
    """
    Source:
        https://github.com/LeCAR-Lab/dial-mpc/blob/main/dial_mpc/utils/io_utils.py#L15
    """
    keys = dataclass.__dataclass_fields__.keys() & data_dict.keys()
    kwargs = {key: data_dict[key] for key in keys}
    if convert_list_to_array:
        import numpy as np

        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = np.array(value)
    return dataclass(**kwargs)


class DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


class MazeGenerator:
    """
    A maze generator that creates a maze with a given width and height.
    The maze is generated using a recursive backtracking algorithm.
    The maze is represented as a 2D list of characters, where '0' represents a wall and '1' represents a cell.

    Args:
        width (int): The width of the maze.
        height (int): The height of the maze.

    Raises:
        ValueError: If the width or height is less than 1.

    Returns:
        list: A 2D list representing the maze.
    """

    def __init__(self, width, height):
        if width < 1 or height < 1:
            raise ValueError("Maze dimensions must be positive")
        self.width = width
        self.height = height
        # Initialize with all walls
        self.maze = [["#" for _ in range(2 * width + 1)] for _ in range(2 * height + 1)]
        # Mark cells (non-wall positions) initially
        for y in range(self.height):
            for x in range(self.width):
                self.maze[2 * y + 1][2 * x + 1] = " "

    def generate(self):
        walls = []
        # Create list of all walls between cells
        for y in range(self.height):
            for x in range(self.width):
                if x < self.width - 1:  # Horizontal walls
                    walls.append((x, y, x + 1, y))
                if y < self.height - 1:  # Vertical walls
                    walls.append((x, y, x, y + 1))

        np.random.shuffle(walls)
        cells = DisjointSet(self.width * self.height)

        # Process walls
        for x1, y1, x2, y2 in walls:
            cell1 = y1 * self.width + x1
            cell2 = y2 * self.width + x2

            if cells.union(cell1, cell2):
                # Remove wall between cells
                wall_x = x1 + x2 + 1
                wall_y = y1 + y2 + 1
                self.maze[wall_y][wall_x] = " "

        # Ensure start and end are accessible
        self.maze[1][0] = " "  # Entrance
        self.maze[-2][-1] = " "  # Exit

        # Validate path exists
        if not self._validate_path():
            return self.generate()  # Regenerate if no valid path

        self.maze = self.convert_maze(self.maze)
        return self.maze

    def _validate_path(self):
        """Validate path exists from start to end using BFS"""
        from collections import deque

        start = (1, 0)  # Entrance coordinates
        end = (len(self.maze) - 2, len(self.maze[0]) - 1)  # Exit coordinates
        visited = set()
        queue = deque([start])

        while queue:
            y, x = queue.popleft()
            if (y, x) == end:
                return True

            for ny, nx in [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]:
                if (
                    0 <= ny < len(self.maze)
                    and 0 <= nx < len(self.maze[0])
                    and self.maze[ny][nx] == " "
                    and (ny, nx) not in visited
                ):
                    visited.add((ny, nx))
                    queue.append((ny, nx))
        return False

    def convert_maze(self, maze):
        return [[("0" if cell == "#" else "1") for cell in row] for row in maze]


def generate_maze(width=3, height=3):
    generator = MazeGenerator(width, height)
    maze = generator.generate()
    return maze
