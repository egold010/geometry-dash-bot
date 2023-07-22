import pygame, pyautogui, numpy as np, gymnasium as gym
from gymnasium.envs.registration import register

register(
     id="GeometryDash-v0",
     entry_point="GeometryDash:GeometryDashEnv",
)

RENDER_MODE = "human"
SIZE = pyautogui.size()
GRID_SIZE = 20

PLAYER_POS = (SIZE[0] * .25, SIZE[1] * .75)
PLAYER_COLOR = (255, 255, 255)

obstacles = [
    "Ground",
    "Spike"
]

class GeometryDashEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self, size = SIZE) -> None:
        self.window_size = size
        self.window = None
        self.clock = None
        self.player = PlayerCube()

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=len(obstacles), shape=(32, 20))

        """temporary initial observation
        later replace with initial state of the level"""
        self.grid = np.zeros((32, 20))

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(SIZE)
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def step(self, action):
        self.render()
        return self.grid, 0, False, False, {}

    def render(self) -> None:
        canvas = pygame.Surface((self.window_size[0], self.window_size[1]))
        canvas.fill((0,0,0))

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
    
    def reset(self):
        return self.grid, {}

class PlayerCube:
    def __init__(self, pos = PLAYER_POS, size = GRID_SIZE, rotation = 0):
        self.X = pos[0]
        self.Y = pos[1]
        self.size = size
        self.rotation = rotation
        
        self.PlayerSquare = pygame.Surface((self.size, self.size))
        self.PlayerSquare.fill(PLAYER_COLOR)
    
    def render(self, window) -> None:
        square = pygame.transform.rotate(self.PlayerSquare, self.rotation)
        rect = square.get_rect()
        rect.center = (self.X, self.Y)
        window.blit(square, rect)