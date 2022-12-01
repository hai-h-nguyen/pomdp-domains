from pdomains import *
import matplotlib.pyplot as plt
import gym

env=gym.make('pdomains-drawer-opening-v0', rendering=True)
obs = env.reset()

cnt = 0
ep_idx = 0
success_ep_idx = 0

while ep_idx <= 100:
    # action = env.action_space.sample()
    action = env.query_expert(ep_idx)
    obs, reward, done, info = env.step(action)
    cnt += 1
    if done:
        if info["success"]:
            success_ep_idx += 1
        print(cnt, success_ep_idx, ep_idx)
        ep_idx += 1
        cnt = 0
        env.reset()
