from pdomains import *
import matplotlib.pyplot as plt
import gym

env=gym.make('pdomains-light-house-v0')
env.reset()

print(env.action_space)
print(env.observation_space)

for i in range(1000):
    action = env.query_expert()
    _, _, done, _ = env.step(action[0])
    env.render()
    if done:
        env.reset()
