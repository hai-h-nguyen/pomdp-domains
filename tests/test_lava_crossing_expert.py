import gym
from pdomains import *
import matplotlib.pyplot as plt

env = gym.make('pdomains-lava-crossing-v0', rendering=True)
env.reset()

step_cnt = 0
ep_cnt = 0

while ep_cnt < 10:
    action = env.query_expert()[0]
    assert (action < 3), action
    step_cnt += 1
    _, reward, done, _ = env.step(action)
    if done:
        print(step_cnt, reward)
        ep_cnt += 1
        env.reset()
        step_cnt = 0