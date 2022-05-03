from gym.envs.registration import register

register(
    id='pdomains-car-flag-v0',
    entry_point='pdomains.car_flag_discrete:CarEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-car-flag-v1',
    entry_point='pdomains.car_flag_discrete_extra:CarEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-light-house-v0',
    entry_point='pdomains.lighthouse:LightHouseEnv',
    max_episode_steps=100,
)

register(
    id='pdomains-lava-crossing-v0',
    entry_point='pdomains.lava_crossing_s25n10:LavaCrossingS25N10Env',
    max_episode_steps=100,
)

register(
    id='pdomains-wall-crossing-v0',
    entry_point='pdomains.wall_crossing_s25n10:WallCrossingS25N10Env',
    max_episode_steps=100,
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
