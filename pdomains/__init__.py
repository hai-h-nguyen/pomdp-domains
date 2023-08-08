from gym.envs.registration import register


register(
    id='peg-insertion-round-real-xyz-v0',
    entry_point='pdomains.peg_insertion_real:PegInsertionEnv',
    max_episode_steps=50,
)

register(
    id='peg-insertion-square-real-xyz-v0',
    entry_point='pdomains.peg_insertion_square_real:PegInsertionEnv',
    max_episode_steps=50,
)
