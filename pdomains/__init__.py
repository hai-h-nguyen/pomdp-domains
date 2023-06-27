from gym.envs.registration import register

register(
    id='block-picking-v0',
    entry_point='pdomains.block_picking:BlockEnv',
    max_episode_steps=50,
)

register(
    id='block-pulling-v0',
    entry_point='pdomains.block_pulling:BlockEnv',
    max_episode_steps=50,
)

register(
    id='block-pushing-v0',
    entry_point='pdomains.block_pushing:BlockEnv',
    max_episode_steps=50,
)

register(
    id='block-stacking-v0',
    entry_point='pdomains.block_stacking:BlockEnv',
    max_episode_steps=50,
)

register(
    id='drawer-opening-v0',
    entry_point='pdomains.drawer_opening:DrawerEnv',
    max_episode_steps=50,
)
