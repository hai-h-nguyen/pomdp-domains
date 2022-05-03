import gym
from pdomains import *
import time

env = gym.make("pdomains-mg-memory-s17-v0")

obs = env.reset()

eps_cnt = 0

while eps_cnt < 10:
    action = env.query_expert()[0]
    _, reward, done, _ = env.step(action)
    env.render()
    time.sleep(0.2)

    if done:
        print("Reset Final Reward:", reward)
        eps_cnt += 1
        env.reset()