from pdomains import *
import gym

# env=gym.make('pdomains-cart-pole-f-v0')
env=gym.make('pdomains-cart-pole-p-v0')
obs = env.reset()
rewards = 0

for i in range(1000):
    action = env.action_space.sample()
    action = env.query_expert()[0]
    obs, reward, done, _ = env.step(action)
    rewards += reward
    env.render()
    if done:
        print(rewards)
        rewards = 0
        env.reset()
