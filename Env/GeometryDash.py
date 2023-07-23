import pygame, pyautogui, numpy as np, gymnasium as gym, math
from gymnasium.envs.registration import register

register(
     id="GeometryDash-v0",
     entry_point="Env.GeometryDash:GeometryDashEnv",
)

RENDER_MODE = "human"
SIZE = pyautogui.size()
GRID_SIZE = SIZE[1] * .05

PLAYER_GRID_POS = (4, 13)
PLAYER_POS = (PLAYER_GRID_POS[0] * GRID_SIZE, PLAYER_GRID_POS[1] * GRID_SIZE)
PLAYER_COLOR = (255, 255, 255)

obstacles = [
    "Ground",
    "Spike"
]

class Obstacle:
    def __init__(self, name, type, geometry, center):
        self.name = name
        self.geometry = geometry
        self.type = type

        if type == "Ground":
            self.hitbox = geometry.get_rect()
            self.hitbox.center = center
        elif type == "Hazard":
            self.hitbox = [
                Utilities.getPointAverage(geometry[0], geometry[2]),
                Utilities.getPointAverage(geometry[1], geometry[2]),
                geometry[2]
            ]

class GeometryDashEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    activeTerrain = []
    
    def __init__(self, size = SIZE) -> None:
        self.window_size = size
        self.window = None
        self.clock = None
        
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=len(obstacles), shape=(32, 20))

        """temporary initial observation
        later replace with initial state of the level"""
        self.grid = np.zeros((32, 20))

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(SIZE)

        self.reset()

    def step(self, action):
        if action:
            self.player.jump()
        fail = self.render()
        reward = -100 if fail else 1
        return self.grid, reward, fail, {}

    def render(self) -> None:
        canvas = pygame.Surface((self.window_size[0], self.window_size[1]))
        canvas.fill((0,0,0))
        self.offset += self.player.x_velocity

        self.drawMap(canvas)
        self.player.update(self.activeTerrain)
        self.player.render(canvas)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

        return not self.player.alive
    
    def reset(self):
        self.player = PlayerCube()
        self.offset = 0
        self.clock = pygame.time.Clock()

        return self.grid, {}

    def close(self) -> None:
        pygame.quit()

    #helpers

    def drawMap(self, window) -> None:
        self.activeTerrain = []
        for y, row in enumerate(levels[0]):
            for x, cell in enumerate(row):
                nx = x * GRID_SIZE - self.offset
                ny = y * GRID_SIZE
                obstacle = None
                if cell == 1:
                    obstacle = Utilities.drawCube((nx,ny), GRID_SIZE, (100,100,100), 0, window)
                elif cell == 2:
                    obstacle = Utilities.drawSpike((nx,ny), GRID_SIZE, (100,50,50), window)
                if obstacle is not None:
                    self.activeTerrain.append(obstacle)
                


class PlayerCube:
    def __init__(self, pos = PLAYER_POS, size = GRID_SIZE, rotation = 0):
        self.X = pos[0]
        self.Y = pos[1]
        self.size = size
        self.rotation = rotation
        self.alive = True

        self.x_velocity = 8
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

    def jump(self) -> None:
        if self.on_ground:  # To prevent double jumps
            self.y_velocity -= self.jump_strength
            self.on_ground = False

    def update(self, terrain) -> None:
        if self.on_ground:
            self.y_velocity = 0
            self.rotation = Utilities.roundToNearest(self.rotation, 90)
        else:
            self.y_velocity += self.gravity
            self.Y += self.y_velocity
            self.rotation += 3

        self.on_ground = False
        for obstacle in terrain:
            r1 = self.PlayerSquare.get_rect()
            r1.center = (self.X, self.Y)
            if obstacle.type == "Ground":
                if r1.colliderect(obstacle.hitbox):
                    self.y_velocity = 0
                    self.on_ground = True
                    self.Y = obstacle.hitbox.top - self.size / 2
                    break
            elif obstacle.type == "Hazard":
                for hb in obstacle.hitbox:
                    if r1.collidepoint(hb):
                        self.alive = False
                        self.x_velocity = 0
                        pyautogui.sleep(3)
                        break

class Utilities:
    @staticmethod
    def drawCube(center, size, color, rotation, window):
        square = pygame.Surface((size, size))
        square.fill(color)
        square = pygame.transform.rotate(square, rotation)
        obstacle = Obstacle("Ground", "Ground", square, center)
        rect = square.get_rect()
        rect.center = center
        if rotation == 0:
            window.blit(square, rect)
        else:
            pass
        return obstacle

    @staticmethod
    def drawSpike(center, size, color, window) -> None:
        p1 = (center[0] - size / 2, center[1] + size / 2)
        p2 = (center[0] + size / 2, center[1] + size / 2)
        p3 = (center[0], center[1] - size / 2)
        pygame.draw.polygon(window, color, [p1, p2, p3])
        obstacle = Obstacle("Spike", "Hazard", [p1, p2, p3], center)
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

"""game is a grid with height 20"""
"""player is at height 13"""

levels = [
    [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ],
]