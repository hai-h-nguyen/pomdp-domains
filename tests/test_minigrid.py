from gym_minigrid.wrappers import *
import time

env = gym.make('MiniGrid-MemoryS17Random-v0')
obs = env.reset() # This noMiniGrid-DoorKey-16x16-v0w produces an RGB tensor only

print(env.action_space.n)

print(obs['image'].shape)

for i in range(1000):
    obs, _, _, _ = env.step(env.action_space.sample())
    print(obs['image'].shape)
    env.render()

    if i % 50 == 0:
        print("Reset")
        env.reset()