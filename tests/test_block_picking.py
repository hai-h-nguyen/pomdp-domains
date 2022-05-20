from pdomains import *
import matplotlib.pyplot as plt
import gym

env=gym.make('pdomains-block-picking-v0', rendering=True)
obs = env.reset()

cnt = 0

for i in range(1000):
    action = env.action_space.sample()
    # print(action)
    action = env.query_expert()
    obs, reward, done, info = env.step(action)
    cnt += 1
    if done:
        print(cnt)
        print(info)
        cnt = 0
        env.reset()
