from gym.envs.registration import register

register(
    id='pdomains-car-flag-v0',
    entry_point='pdomains.car_flag_discrete:CarEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-car-flag-v1',
    entry_point='pdomains.car_flag_discrete_additional:CarEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-car-flag-pomdp-v0',
    entry_point='pdomains.car_flag_pomdp:CarEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-poisoned-doors-v0',
    entry_point='pdomains.poisoned_doors:PoisonedDoorsEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-car-flag-big-mdp-v0',
    entry_point='pdomains.car_flag_big_mdp:CarEnv',
    max_episode_steps=250,
)

register(
    id='pdomains-two-boxes-v0',
    entry_point='pdomains.two_boxes_discrete:BoxEnv',
    max_episode_steps=100,
)

register(
    id='pdomains-two-boxes-mdp-v0',
    entry_point='pdomains.two_boxes_mdp:BoxEnv',
    max_episode_steps=100,
)

register(
    id='pdomains-ant-heaven-hell-v0',
    entry_point='pdomains.ant_heaven_hell:AntEnv',
    max_episode_steps=400,
)

register(
    id='pdomains-ant-tag-v0',
    entry_point='pdomains.ant_tag:AntTagEnv',
    max_episode_steps=400,
)

register(
    id='pdomains-light-house-1D-v0',
    entry_point='pdomains.lighthouse_1D:LightHouse1DEnv',
    max_episode_steps=100,
)

register(
    id='pdomains-light-house-v0',
    entry_point='pdomains.lighthouse:LightHouseEnv',
    max_episode_steps=100,
)