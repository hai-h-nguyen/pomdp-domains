from gym.envs.registration import register


# POMDPs
register(
    id='peg-insertion-square-xz-v0',
    entry_point='pdomains.peg_insertion_xz:PegInsertionEnv',
    max_episode_steps=150,
    kwargs={"peg_type": "square"}
)

register(
    id='peg-insertion-square-xyz-v0',
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "square"}
)

register(
    id='peg-insertion-oblong-xyz-v0',
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "oblong"}
)

register(
    id='peg-insertion-triangle-xyz-v0',
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "triangle"}
)


# State version
register(
    id='peg-insertion-square-state-xz-v0',
    entry_point='pdomains.peg_insertion_xz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "square", "return_state": True}
)

register(
    id='peg-insertion-square-state-xyz-v0',
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "square", "return_state": True}
)
