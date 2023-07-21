from gym.envs.registration import register


register(
    id='peg-insertion-triangle-xyz-v0',
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "triangle"}
)

register(
    id='peg-insertion-square-xyz-v0',
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "square"}
)

register(
    id='peg-insertion-square-old-xyz-v0',
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "square-old"}
)

register(
    id='peg-insertion-pentagon-xyz-v0',
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "pentagon"}
)

register(
    id='peg-insertion-hexagon-xyz-v0',
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "hexagon"}
)

register(
    id='peg-insertion-round-xyz-v0',
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "round"}
)
