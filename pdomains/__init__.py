from gym.envs.registration import register

register(
    id='pdomains-car-flag-symm-v1',
    entry_point='pdomains.symmetric_car_flag_discrete_1d:CarEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-car-flag-asymm-v0',
    entry_point='pdomains.asymmetric_car_flag_discrete_1d:CarEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-car-flag-symm-continuous-v0',
    entry_point='pdomains.symmetric_car_flag_continuous_1d:CarEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-car-flag-symm-2d-v0',
    entry_point='pdomains.symmetric_car_flag_discrete_2d:CarEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-block-picking-v0',
    entry_point='pdomains.block_picking:BlockEnv',
    max_episode_steps=50,
)
