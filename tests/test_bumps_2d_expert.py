from pdomains import *
import gym

env=gym.make('pdomains-bumps-2d-v0', rendering=True)
env.reset()

for i in range(100000):
    action = env.query_expert()[0]
    print(action)
    _, reward, done, _ = env.step(action)
    if done:
        print("Done:", reward)
        env.reset()
