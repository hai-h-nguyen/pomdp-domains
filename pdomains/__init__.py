from gym.envs.registration import register

register(
    id='pdomains-peg-insertion-square-v0',
    entry_point='pdomains.peg_insertion:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "square"}
)


register(
    id='pdomains-peg-insertion-square-xz-v0',
    entry_point='pdomains.peg_insertion_xz:PegInsertionEnv',
    max_episode_steps=150,
    kwargs={"peg_type": "square"}
)


register(
    id='pdomains-peg-insertion-hex-star-v0',
    entry_point='pdomains.peg_insertion:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "hex-star"}
)

register(
    id='pdomains-peg-insertion-state-square-v0',
    entry_point='pdomains.peg_insertion_state:PegInsertionEnv',
    max_episode_steps=150,
    kwargs={"peg_type": "square"}
)

register(
    id='pdomains-peg-insertion-state-hex-star-v0',
    entry_point='pdomains.peg_insertion_state:PegInsertionEnv',
    max_episode_steps=100,
    kwargs={"peg_type": "hex-star"}
)
