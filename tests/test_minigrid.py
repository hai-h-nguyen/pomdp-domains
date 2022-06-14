from gym_minigrid.wrappers import *
import time

env = gym.make('MiniGrid-MemoryS17Random-v0')
obs = env.reset() # This noMiniGrid-DoorKey-16x16-v0w produces an RGB tensor only

test_episodes = 100
eps_cnt = 0

for i in range(test_episodes):
    obs, _, _, _ = env.step(env.action_space.sample())
    env.render()

    if i % 50 == 0:
        print("Reset")
        env.reset()