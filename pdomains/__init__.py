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
    id='pdomains-two-boxes-v0',
    entry_point='pdomains.two_boxes_discrete:BoxEnv',
    max_episode_steps=100,
)

register(
    id='pdomains-light-house-v0',
    entry_point='pdomains.lighthouse:LightHouseEnv',
    max_episode_steps=100,
)