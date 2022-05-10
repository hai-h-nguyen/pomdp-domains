import gym
import numpy as np

from stable_baselines3 import SAC
import pybullet_envs

env = gym.make("HalfCheetahBulletEnv-v0")

model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000000, log_interval=4)
# model.save("sac_cheetah")

# del model # remove to demonstrate saving and loading

model = SAC.load("sac_cheetah")

obs = env.reset()
rewards = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    rewards += reward
    env.render()
    if done:
      obs = env.reset()
      print(rewards)
      rewards = 0