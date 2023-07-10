from gym.envs.registration import register


register(
    id='peg-insertion-round-xyz-v0',
    entry_point='pdomains.peg_insertion_real:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "round"}
)
