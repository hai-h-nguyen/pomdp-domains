import gym
from pdomains import *
import time

env = gym.make('pdomains-wall-crossing-v0', rendering=True)
env.reset()

print(env.action_space)

step_cnt = 0
ep_cnt = 0

while ep_cnt < 5:
    action = env.query_expert()[0]
    assert (action < 4), action
    step_cnt += 1
    _, reward, done, _ = env.step(action)
    time.sleep(0.1)
    if done:
        print(step_cnt, reward)
        ep_cnt += 1
        env.reset()
        step_cnt = 0