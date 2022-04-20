from pdomains import *
import matplotlib.pyplot as plt
import gym

env=gym.make('pdomains-light-house-v0')
env.reset()

print(env.action_space)
print(env.observation_space)

for i in range(1000):
    action = env.action_space.sample()
    env.step(action)
    env.render()
    # print(env.query_expert())
    if i % 10 == 0:
        env.reset()
        # print("Reset:", env.reset())
