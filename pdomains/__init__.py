from gym.envs.registration import register


# POMDPs
register(
    id='peg-insertion-square-xz-v0',
    entry_point='pdomains.peg_insertion_xz:PegInsertionEnv',
    max_episode_steps=150,
    kwargs={"peg_type": "square"}
)

register(
    id='peg-insertion-square-xyz-no-torques-v0',
    entry_point='pdomains.peg_insertion_xyz_no_torques:PegInsertionEnv',
    max_episode_steps=150,
    kwargs={"peg_type": "square"}
)

register(
    id='peg-insertion-square-xyz-v0',
    entry_point='pdomains.peg_insertion_xyz_no_torques:PegInsertionEnv',
    max_episode_steps=150,
    kwargs={"peg_type": "square", "torques": True}
)
