import gymnasium as gym
import Env.GeometryDash as GeometryDash
import stable_baselines3 as sb3
import math

env = gym.make("GeometryDash-v0")
model = sb3.PPO("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=10, log_interval=math.inf)

#model_name = "ppo-LunarLander-v2"
#model.save(model_name)

for _ in range(1000):
    observation, info = env.reset(options={"Render": True})
    action = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated:
        observation, info = env.reset(options={"Render": True})

env.close()