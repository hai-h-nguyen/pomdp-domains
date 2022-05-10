from pdomains import *
import gym

env=gym.make('pdomains-half-cheetah-p-v0', rendering=True)
obs = env.reset()
returns = 0

for i in range(10000):
    action = env.query_expert()[0]
    obs, reward, done, info = env.step(action)
    returns += reward
    if done:
        print(returns)
        returns = 0
        env.reset()
