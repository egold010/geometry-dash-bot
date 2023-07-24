import gymnasium as gym
import Env.GeometryDash as GeometryDash
import stable_baselines3 as sb3
import math

env = gym.make("GeometryDash-v0")
model = sb3.PPO("MlpPolicy", env, verbose=1)
model = model.learn(total_timesteps=20480)
GeometryDash.do_render = True
print("rendering")
model = model.learn(total_timesteps=2048000)

#model_name = "ppo-LunarLander-v2"
#model.save(model_name)

observation, info = env.reset(options={"Render": True})

for _ in range(int(1e12)):
    action = model.predict(observation, deterministic=False)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated:
        observation, info = env.reset(options={"Render": True})

env.close()