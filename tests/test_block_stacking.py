from pdomains import *
import matplotlib.pyplot as plt
import gym

env=gym.make('pdomains-block-stacking-v0', rendering=True)
obs = env.reset()

cnt = 0
ep_idx = 0

for i in range(1000):
    # action = env.action_space.sample()
    action = env.query_expert(ep_idx)
    obs, reward, done, info = env.step(action)
    cnt += 1
    if done:
        print(cnt, info, ep_idx)
        ep_idx += 1
        cnt = 0
        env.reset()
