from pdomains import *
import gym
import numpy as np

# env=gym.make('pdomains-mg-memory-s17-v0', rendering=False)
# env=gym.make('pdomains-car-flag-continuous-v0', rendering=False)
# env=gym.make('pdomains-half-cheetah-v-v0', rendering=False)
# env=gym.make('pdomains-lunar-lander-v-v0', rendering=False)
env=gym.make('pdomains-block-picking-v0', rendering=False)
env.reset()

num_eps = 100
ep_cnt = 0
success = []
rewards = []
ep_reward = 0

while (ep_cnt < num_eps):
    action = env.action_space.sample()
    # action = env.query_expert()[0]
    _, reward, done, info = env.step(action)
    ep_reward += reward
    if done:
        ep_cnt += 1
        rewards.append(ep_reward)
        ep_reward = 0
        success.append(info["success"])
        env.reset()

print("Success rate:", np.mean(success), np.std(success))
print("Reward:", np.mean(rewards), np.std(rewards))
