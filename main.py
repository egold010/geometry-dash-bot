import signal, sys
import gymnasium as gym
import Env.GeometryDash as GeometryDash
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

#region Signal Handler
def signal_handler(sig, frame):
    print('You pressed Ctrl+C or stopped the program from your IDE!')
    env.close()
    sys.exit(0)

# Register the signal handler for termination signals
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
#endregion

#region Environment
TRAIN = False
GeometryDash.Manual = True
#try:
env = DummyVecEnv([lambda: gym.make("GeometryDash-v0") for _ in range(8)])
#env = Monitor(gym.make("GeometryDash-v0"), "./GD_logs")
model = sb3.PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.00025,
    n_steps=2048,
    batch_size=128,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    verbose=1,
    #tensorboard_log="./tensorboard_logs"
)

if TRAIN:
    model.learn(total_timesteps=10000000, progress_bar=True)
    model.save("Models/GeometryDash-v0")
else:
    del model
    model = sb3.PPO.load("Models/GeometryDash-v0")

del env
env = gym.make("GeometryDash-v0")
observation, info = env.reset()
for _ in range(int(1e12)):
    action, _steps = model.predict(observation, deterministic=True)
    observation, reward, done, truncated, info = env.step(action)
    env.render()

    if done or truncated:
        observation, info = env.reset()

#finally:
    #env.close()
#endregion