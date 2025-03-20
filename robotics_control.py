import numpy as np
import random
import matplotlib.pyplot as plt

class Robot:
    def __init__(self, x, y, orientation):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.obstacles = []
        self.path = []

    def move_forward(self, distance):
        if self.orientation == 'N':
            self.y += distance
        elif self.orientation == 'S':
            self.y -= distance
        elif self.orientation == 'E':
            self.x += distance
        elif self.orientation == 'W':
            self.x -= distance
        self.path.append((self.x, self.y))

    def turn_left(self):
        orientations = ['N', 'W', 'S', 'E']
        index = (orientations.index(self.orientation) + 1) % 4
        self.orientation = orientations[index]

    def turn_right(self):
        orientations = ['N', 'E', 'S', 'W']
        index = (orientations.index(self.orientation) + 1) % 4
        self.orientation = orientations[index]

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def detect_obstacles(self):
        for obs in self.obstacles:
            if (self.x, self.y) == obs:
                print(f"Obstacle detected at {obs}!")

    def plot(self):
        plt.plot(self.x, self.y, 'ro')
        for obs in self.obstacles:
            plt.plot(obs[0], obs[1], 'bo')
        for point in self.path:
            plt.plot(point[0], point[1], 'go')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.grid()
        plt.title("Robot's Path and Obstacles")
        plt.show()

class Environment:
    def __init__(self, width, height, num_obstacles):
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.obstacles = self.generate_obstacles()

    def generate_obstacles(self):
        obstacles = []
        while len(obstacles) < self.num_obstacles:
            obs = (random.randint(-self.width//2, self.width//2), random.randint(-self.height//2, self.height//2))
            if obs not in obstacles:
                obstacles.append(obs)
        return obstacles

    def get_obstacles(self):
        return self.obstacles

def main():
    env = Environment(20, 20, 10)
    robot = Robot(0, 0, 'N')
    robot.set_obstacles(env.get_obstacles())

    commands = ['F', 'R', 'F', 'F', 'L', 'F', 'F', 'R', 'F']
    for cmd in commands:
        if cmd == 'F':
            robot.move_forward(1)
        elif cmd == 'L':
            robot.turn_left()
        elif cmd == 'R':
            robot.turn_right()
        robot.detect_obstacles()

    robot.plot()

if __name__ == "__main__":
    main()

# Adding more features: Dynamic obstacle avoidance and planning
class AdvancedRobot(Robot):
    def __init__(self, x, y, orientation):
        super().__init__(x, y, orientation)

    def dynamic_avoidance(self):
        for obs in self.obstacles:
            if abs(obs[0] - self.x) < 1 and abs(obs[1] - self.y) < 1:
                print(f"Avoiding obstacle at {obs}")
                self.turn_left()
                self.move_forward(2)

    def advanced_plot(self):
        plt.plot(self.x, self.y, 'ro', markersize=12)
        for obs in self.obstacles:
            plt.plot(obs[0], obs[1], 'bo', markersize=12)
        for point in self.path:
            plt.plot(point[0], point[1], 'go')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.grid()
        plt.title("Advanced Robot's Path and Obstacles")
        plt.show()

def advanced_main():
    env = Environment(20, 20, 10)
    advanced_robot = AdvancedRobot(0, 0, 'N')
    advanced_robot.set_obstacles(env.get_obstacles())

    commands = ['F', 'R', 'F', 'F', 'L', 'F', 'F', 'R', 'F']
    for cmd in commands:
        if cmd == 'F':
            advanced_robot.move_forward(1)
        elif cmd == 'L':
            advanced_robot.turn_left()
        elif cmd == 'R':
            advanced_robot.turn_right()
        advanced_robot.dynamic_avoidance()

    advanced_robot.advanced_plot()

if __name__ == "__main__":
    advanced_main()

# Adding pathfinding capability
class Pathfinder:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        self.grid = np.zeros((grid_size, grid_size))

    def set_obstacles(self, obstacles):
        for obs in obstacles:
            self.grid[obs[0]][obs[1]] = 1

    def a_star(self):
        # Implementation of A* Algorithm
        open_set = {self.start}
        came_from = {}

        g_score = {point: float('inf') for point in self.get_all_points()}
        g_score[self.start] = 0
        f_score = {point: float('inf') for point in self.get_all_points()}
        f_score[self.start] = self.heuristic(self.start)

        while open_set:
            current = min(open_set, key=lambda point: f_score[point])
            if current == self.goal:
                return self.reconstruct_path(came_from, current)

            open_set.remove(current)
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor)
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return []

    def get_neighbors(self, point):
        neighbors = []
        x, y = point
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if abs(dx) != abs(dy):  # not diagonal
                    new_point = (x + dx, y + dy)
                    if self.is_valid(new_point):
                        neighbors.append(new_point)
        return neighbors

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def heuristic(self, point):
        return abs(point[0] - self.goal[0]) + abs(point[1] - self.goal[1])

    def is_valid(self, point):
        x, y = point
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.grid[x][y] == 0

    def get_all_points(self):
        return [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]

def pathfinding_main():
    env = Environment(20, 20, 10)
    pathfinder = Pathfinder(20)
    pathfinder.set_obstacles(env.get_obstacles())

    path = pathfinder.a_star()
    print("Calculated Path:", path)

if __name__ == "__main__":
    pathfinding_main()

# Infusing learning components to the robot
class LearningRobot(Robot):
    def __init__(self, x, y, orientation):
        super().__init__(x, y, orientation)
        self.learned_paths = []

    def learn_from_path(self):
        self.learned_paths.append(self.path.copy())
        print("Path learned:", self.path)

    def recall_path(self, index):
        if index < len(self.learned_paths):
            print("Recalling path:", self.learned_paths[index])
            for coord in self.learned_paths[index]:
                self.x, self.y = coord
                print(f"Moved to {coord}")

def learning_main():
    env = Environment(20, 20, 5)
    learning_robot = LearningRobot(0, 0, 'N')
    learning_robot.set_obstacles(env.get_obstacles())

    commands = ['F', 'F', 'L', 'F', 'R', 'F', 'F']
    for cmd in commands:
        if cmd == 'F':
            learning_robot.move_forward(1)
        elif cmd == 'L':
            learning_robot.turn_left()
        elif cmd == 'R':
            learning_robot.turn_right()

    learning_robot.learn_from_path()
    learning_robot.recall_path(0)

if __name__ == "__main__":
    learning_main()