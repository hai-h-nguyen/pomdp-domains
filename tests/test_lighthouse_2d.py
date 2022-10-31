from pdomains import *
import matplotlib.pyplot as plt
import gym

env=gym.make('pdomains-light-house-2d-v0')
env.reset()

print(env.action_space)
print(env.observation_space)

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if i % 10 == 0:
        env.reset()
