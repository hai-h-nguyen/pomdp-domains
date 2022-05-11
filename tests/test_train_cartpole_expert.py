import gym
import numpy as np
from pdomains import *

from stable_baselines3 import SAC

env = gym.make("pdomains-cart-pole-f-v0")

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000, log_interval=4)
model.save("sac_cartpole")

# del model # remove to demonstrate saving and loading

# model = SAC.load("sac_cart_pole")

# obs = env.reset()
# print(obs.shape)
# rewards = 0
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     rewards += reward
#     env.render()
#     if done:
#       obs = env.reset()
#       print(rewards)
#       rewards = 0