import gymnasium as gym
import Env.GeometryDash as GeometryDash

env = gym.make("GeometryDash-v0")
observation, info = env.reset()

for _ in range(1000):
    #action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step("""action""")

    """if terminated or truncated:
        observation, info = env.reset()"""

env.close()