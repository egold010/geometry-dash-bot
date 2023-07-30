import pygame, pyautogui, numpy as np, gymnasium as gym, math, random
from gymnasium.envs.registration import register

register(
     id="GeometryDash-v0",
     entry_point="Env.GeometryDash:GeometryDashEnv",
)

RENDER_MODE = "human"
SIZE = (1440, 900)
GRID_SIZE = SIZE[1] * .05

OBSERVATION_CELL_SIZE = 20#10
OBSERVATION_DIMENSION = (round(SIZE[0]/OBSERVATION_CELL_SIZE), round(SIZE[1]/OBSERVATION_CELL_SIZE))

PLAYER_GRID_POS = (4, 13)
PLAYER_POS = (PLAYER_GRID_POS[0] * GRID_SIZE, PLAYER_GRID_POS[1] * GRID_SIZE)
PLAYER_COLOR = (255, 255, 255)

obstacles = [
    "Air",
    "Ground",
    "Spike",
    "Half Spike",
    "Player",
    "Reward"
]
modes = [
    "Ground",
    "Ship"
]

Manual = False

class GeometryDashEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    activeTerrain = []
    
    def __init__(self, size = SIZE) -> None:
        self.window_size = size
        self.window = None
        self.clock = None
        self.total_reward = 0
        self.done = False
        self.level = levels[0]
        
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=len(obstacles), shape=(OBSERVATION_DIMENSION[0] * OBSERVATION_DIMENSION[1],))

        self.reset()

    def step(self, action):
        if Manual:
            action = pygame.mouse.get_pressed()[0]

        self.player.action(action)
        
        self.setMap()
        grid, grid2d = self.getGrid()
        bonus = self.player.update(grid2d)
        fail = not self.player.alive

        reward = 0 #-1000 if fail else 1
        if bonus:
            reward += 0.5
        if fail:
            reward -= 10
        if self.player.alive:
            reward += 0.1
        if action:
            reward -= 0.2
        if self.player.Y > SIZE[1] * .75:
            reward += 100

        self.offset += self.player.x_velocity
        self.total_reward += reward
        self.done = fail

        return grid, reward, fail, self.player.Y > SIZE[1] * .75, {}

    def render(self) -> None:
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(SIZE)

        canvas = pygame.Surface((self.window_size[0], self.window_size[1]))
        canvas.fill((0,0,0))

        self.drawMap(canvas)
        self.player.render(canvas)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
    
    def reset(self, seed = None, options = None):
        if options is not None:
            do_render = options["Render"]

        self.player = PlayerCube()
        self.offset = 0
        self.clock = pygame.time.Clock()

        #TEMP
        print("Reward:", self.total_reward)
        """eff_row = 13
        col = random.randint(10, len(self.level[eff_row]) - 1)
        col2 = col
        while abs(col2 - col) < 4:
            col2 = random.randint(10, len(self.level[eff_row]) - 1)

        for i in range(len(self.level[eff_row])):
            self.level[eff_row][i] = 2 if i == col or i == col2 else 0"""
        self.level = levels[random.randint(0, len(levels) - 1)]

        self.total_reward = 0
        grid, grid2d = self.getGrid()
        return grid, {}

    def close(self) -> None:
        pygame.quit()

    #helpers

    def setMap(self) -> None:
        self.activeTerrain = []
        for y, row in enumerate(self.level):
            for x, cell in enumerate(row):
                nx = x * GRID_SIZE - self.offset
                ny = y * GRID_SIZE
                obstacle = None
                if nx > 0 - GRID_SIZE and nx < SIZE[0] + GRID_SIZE and cell != 0:
                    if cell == 1:
                        obstacle = Utilities.getCube((nx,ny), GRID_SIZE, (100,100,100), 0)
                    elif cell <= 3:
                        obstacle = Utilities.getSpike((nx,ny), GRID_SIZE, (100,50,50), cell == 3)
                    elif cell == 5:
                        obstacle = Obstacle("Reward", "Reward", pygame.surface.Surface((GRID_SIZE, GRID_SIZE)), (nx, ny))
                    self.activeTerrain.append(obstacle)

    def drawMap(self, window) -> None:
        for obstacle in self.activeTerrain:
            obstacle.draw(window)
                
    def getGrid(self): #todo create second grid that stores the obstacle object instead of the type and stays 2d
        f = OBSERVATION_CELL_SIZE
        grid = np.zeros((round(SIZE[0]/f), round(SIZE[1]/f)))
        grid2d = [[0]*round(SIZE[0]/f) for _ in range(round(SIZE[1]/f))]
        for obstacle in self.activeTerrain:
            if obstacle.type == "Ground":
                if round(obstacle.hitbox.left/f) < 0 or round(obstacle.hitbox.right/f) >= round(SIZE[0]/f) or round(obstacle.hitbox.top/f) < 0 or round(obstacle.hitbox.bottom/f) >= round(SIZE[1]/f):
                    continue
                grid[round(obstacle.hitbox.left/f):round(obstacle.hitbox.right/f), round(obstacle.hitbox.top/f):round(obstacle.hitbox.bottom/f)] = 1
                Utilities.setArrSlice(grid2d, round(obstacle.hitbox.top/f), round(obstacle.hitbox.bottom/f), round(obstacle.hitbox.left/f), round(obstacle.hitbox.right/f), obstacle)
            elif obstacle.type == "Hazard":
                if round(obstacle.center[0]/f) < 0 or round(obstacle.center[0]/f) >= round(SIZE[0]/f):
                    continue
                grid[round(obstacle.center[0]/f), round(obstacle.center[1]/f)] = 2
                grid2d[round(obstacle.center[1]/f)][round(obstacle.center[0]/f)] = obstacle
            elif obstacle.type == "Reward":
                if round(obstacle.hitbox.left/f) < 0 or round(obstacle.hitbox.right/f) >= round(SIZE[0]/f) or round(obstacle.hitbox.top/f) < 0 or round(obstacle.hitbox.bottom/f) >= round(SIZE[1]/f):
                    continue
                Utilities.setArrSlice(grid2d, round(obstacle.hitbox.top/f), round(obstacle.hitbox.bottom/f), round(obstacle.hitbox.left/f), round(obstacle.hitbox.right/f), obstacle)
        
        grid[round(self.player.X/f), round(self.player.Y/f)] = 4
        #Utilities.printGrid(grid)
        return np.array(grid, dtype=np.float32).flatten(), grid2d

class PlayerCube:
    ground_gravity = 1
    ship_gravity = .5

    def __init__(self, pos = PLAYER_POS, size = GRID_SIZE, rotation = 0):
        self.X = pos[0]
        self.Y = pos[1]
        self.size = size
        self.rotation = rotation
        self.alive = True

        self.mode = "Ship"
        self.x_velocity = 7
        self.y_velocity = 0
        self.gravity = 1
        self.jump_strength = 15
        self.on_ground = True
        
        self.PlayerSquare = pygame.Surface((self.size, self.size))
        self.PlayerSquare.fill(PLAYER_COLOR)
    
    def render(self, window) -> None:
        """square = pygame.transform.rotate(self.PlayerSquare, self.rotation)
        rect = square.get_rect()
        rect.center = (self.X, self.Y)
        window.blit(square, rect)""" #potentially works with image instead of surface
        Utilities.drawRotatedSquare(window, (self.X, self.Y), self.size, self.rotation, PLAYER_COLOR)

    def action(self, value) -> None:
        if self.alive:
            if self.mode == "Ground":
                if self.on_ground and value:
                    self.y_velocity -= self.jump_strength
                    self.on_ground = False
            if self.mode == "Ship":
                self.gravity = -self.ship_gravity if value else self.ship_gravity
            

    def update(self, terrain) -> None:
        if self.mode == "Ground":
            self.gravity = self.ground_gravity
            if self.on_ground:
                self.y_velocity = 0
                self.rotation = Utilities.roundToNearest(self.rotation, 90)
            else:
                self.y_velocity += self.gravity
                self.Y += self.y_velocity
                self.rotation += 3
        elif self.mode == "Ship":
            self.y_velocity += self.gravity

        bonus = False
        self.on_ground = False

        r1 = self.PlayerSquare.get_rect()
        r1.center = (self.X, self.Y)
        f = OBSERVATION_CELL_SIZE
        squares = [item for sublist in [row[round(r1.left/f)-1:round(r1.right/f)+1] for row in terrain[round(r1.top/f)-1:round(r1.bottom/f)+1]] for item in sublist]

        for obstacle in squares:
            if obstacle != 0:
                if obstacle.type == "Ground":
                    if r1.colliderect(obstacle.hitbox):
                        if r1.bottom <= obstacle.hitbox.top + 1 + self.y_velocity: #buffer
                            self.y_velocity = 0
                            self.on_ground = True
                            self.Y = obstacle.hitbox.top - self.size / 2
                            break
                        else:
                            self.alive = False
                            self.x_velocity = 0
                            break
                elif obstacle.type == "Hazard":
                    for hb in obstacle.hitbox:
                        if r1.collidepoint(hb):
                            self.alive = False
                            self.x_velocity = 0
                            break
                elif obstacle.type == "Reward":
                    if r1.colliderect(obstacle.hitbox):
                        bonus = True
        
        return bonus and self.alive


class Obstacle:
    def __init__(self, name, type, geometry, center):
        self.name = name
        self.geometry = geometry
        self.type = type
        self.center = center

        if type == "Ground" or type == "Reward":
            self.hitbox = geometry.get_rect()
            self.hitbox.center = center
        elif type == "Hazard":
            self.hitbox = [
                Utilities.getPointAverage(geometry[0], geometry[2]),
                Utilities.getPointAverage(geometry[1], geometry[2]),
                (geometry[2][0], geometry[2][1] - 10)
            ]

    def draw(self, canvas):
        if self.type == "Ground":
            rect = self.geometry.get_rect()
            rect.center = self.hitbox.center
            canvas.blit(self.geometry, rect)
        elif self.type == "Hazard":
            pygame.draw.polygon(canvas, self.color, self.geometry)

class Utilities:
    @staticmethod
    def getCube(center, size, color, rotation):
        square = pygame.Surface((size, size))
        square.fill(color)
        square = pygame.transform.rotate(square, rotation)
        obstacle = Obstacle("Ground", "Ground", square, center)
        return obstacle

    @staticmethod
    def getSpike(center, size, color, half = False):
        p1 = (center[0] - size / 2, center[1] + size / 2)
        p2 = (center[0] + size / 2, center[1] + size / 2)
        p3 = center if half else (center[0], center[1] - size / 2)
        obstacle = Obstacle("Spike", "Hazard", [p1, p2, p3], center)
        obstacle.color = color
        return obstacle
    
    @staticmethod
    def roundToNearest(n, i):
        return round(n / i) * i

    @staticmethod
    def rotatePoint(origin, point, angle):
        angle *= math.pi / 180

        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

        return qx, qy

    @staticmethod
    def drawRotatedSquare(window, origin, size, angle, color):
        half = size / 2
        points = [
            (origin[0] - half, origin[1] - half),  # Top-left
            (origin[0] + half, origin[1] - half),  # Top-right
            (origin[0] + half, origin[1] + half),  # Bottom-right
            (origin[0] - half, origin[1] + half),  # Bottom-left
        ]

        # Rotate all points
        points = [Utilities.rotatePoint(origin, point, angle) for point in points]

        pygame.draw.polygon(window, color, points)

    @staticmethod
    def getPointAverage(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    
    @staticmethod
    def printGrid(grid):
        print("Grid:")
        for row in grid:
            for cell in row:
                print(int(cell), end=",")
            print("")

    def setArrSlice(lst, a, b, c, d, val):
        for i in range(a, b):
            for j in range(c, d):
                lst[i][j] = val

levels = [
    [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,2,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,1,3,3,3,1,3,3,3,1,0,0,0,0,0,0,0,0,],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
    ],
    [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,1,1,1,1,1,1,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,3,3,3,1,1,1,1,1,1,1,1,1,1,1,3,3,3,1,1,1,1,1,1,],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
    ],
    [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
    ],
]