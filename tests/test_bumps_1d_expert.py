from pdomains import *
import gym

env=gym.make('pdomains-bumps-1d-v0', rendering=True)
env.reset()

for i in range(1000):
    # action = env.action_space.sample()
    action = env.query_expert()[0]
    _, _, done, _ = env.step(action)
    if done:
        env.reset()