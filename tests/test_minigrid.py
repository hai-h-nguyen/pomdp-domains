from gym_minigrid.wrappers import *

env = gym.make('MiniGrid-Empty-8x8-v0')
obs = env.reset() # This now produces an RGB tensor only

print(env.action_space.n)

print(obs['image'].shape[:2])