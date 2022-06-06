from pdomains import *
import matplotlib.pyplot as plt
import gym

env=gym.make('pdomains-heaven-hell-v0', rendering=False)
obs = env.reset()

cnt = 0

for i in range(100):
    action = env.action_space.sample()
    action = env.query_expert()[0]
    obs, reward, done, info = env.step(action)
    cnt += 1
    if done:
        print(reward, "Done")
        cnt = 0
        env.reset()
