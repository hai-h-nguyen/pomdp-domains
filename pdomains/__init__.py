from gym.envs.registration import register

register(
    id='pdomains-car-flag-continuous-v0',
    entry_point='pdomains.car_flag_continuous:CarEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-bumps-1d-v0',
    entry_point='pdomains.bumps_1d:Bumps1DEnv',
    max_episode_steps=100,
)

register(
    id='pdomains-bumps-2d-v0',
    entry_point='pdomains.bumps_2d:Bumps2DEnv',
    max_episode_steps=100,
)

register(
    id='pdomains-mg-memory-s17-v0',
    entry_point='pdomains.mg_memory_s17:MemoryS17Env',
    max_episode_steps=100,
)

# observation shape is permuted 7, 7, 3 --> 3, 7, 7
register(
    id='pdomains-mg-memory-s17-v1',
    entry_point='pdomains.mg_memory_s17_reshaped:MemoryS17Env',
    max_episode_steps=100,
)


register(
    id='pdomains-lunar-lander-p-v0',
    entry_point='pdomains.lunarlander_p:LunarLanderEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-lunar-lander-v-v0',
    entry_point='pdomains.lunarlander_v:LunarLanderEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-block-picking-v0',
    entry_point='pdomains.block_picking:BlockEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-block-picking-v1',
    entry_point='pdomains.block_picking_dev:BlockEnv',
    max_episode_steps=50,
)