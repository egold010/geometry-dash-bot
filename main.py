import gymnasium as gym
import Env.GeometryDash as GeometryDash

env = gym.make("GeometryDash-v0")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, info = env.step(action)

    if terminated:
        observation, info = env.reset()

env.close()