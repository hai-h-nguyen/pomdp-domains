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
    id='pdomains-car-flag-symm-v0',
    entry_point='pdomains.symmetric_car_flag_discrete:CarEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-car-flag-symm-v1',
    entry_point='pdomains.symmetric_car_flag_discrete_big:CarEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-car-flag-continuous-v0',
    entry_point='pdomains.car_flag_continuous:CarEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-car-flag-continuous-symm-v0',
    entry_point='pdomains.symmetric_car_flag_continuous:CarEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-light-house-v0',
    entry_point='pdomains.lighthouse:LightHouseEnv',
    max_episode_steps=100,
)

register(
    id='pdomains-light-house-2d-v0',
    entry_point='pdomains.lighthouse_2d:LightHouseEnv',
    max_episode_steps=100,
)

register(
    id='pdomains-lava-crossing-v0',
    entry_point='pdomains.lava_crossing_s25n10:LavaCrossingS25N10Env',
    max_episode_steps=100,
)

register(
    id='pdomains-wall-crossing-v0',
    entry_point='pdomains.wall_crossing_s25n10:WallCrossingS25N10Env',
    max_episode_steps=100,
)

register(
    id='pdomains-bumps-1d-v0',
    entry_point='pdomains.bumps_1d:Bumps1DEnv',
    max_episode_steps=100,
)

register(
    id='pdomains-bumps-2d-v0',
    entry_point='pdomains.bumps_2d:Bumps2DEnv',
    max_episode_steps=100,
)

register(
    id='pdomains-bumps-2d-penalty-v0',
    entry_point='pdomains.bumps_2d_penalty:Bumps2DEnv',
    max_episode_steps=100,
)

register(
    id='pdomains-mg-memory-s17-v0',
    entry_point='pdomains.mg_memory_s17:MemoryS17Env',
    max_episode_steps=100,
)

# observation shape is permuted 7, 7, 3 --> 3, 7, 7
register(
    id='pdomains-mg-memory-s17-v1',
    entry_point='pdomains.mg_memory_s17_reshaped:MemoryS17Env',
    max_episode_steps=100,
)


register(
    id='pdomains-half-cheetah-p-v0',
    entry_point='pdomains.half_cheetah_p:HalfCheetahEnv',
    max_episode_steps=200,
)

register(
    id='pdomains-half-cheetah-v-v0',
    entry_point='pdomains.half_cheetah_v:HalfCheetahEnv',
    max_episode_steps=200,
)

register(
    id='pdomains-lunar-lander-p-v0',
    entry_point='pdomains.lunarlander_p:LunarLanderEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-lunar-lander-v-v0',
    entry_point='pdomains.lunarlander_v:LunarLanderEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-cart-pole-p-v0',
    entry_point='pdomains.cartpole_p:CartPolePEnv',
    max_episode_steps=160,
)

# fully observable version of cart pole
register(
    id='pdomains-cart-pole-f-v0',
    entry_point='pdomains.cartpole:CartPoleEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-block-picking-v0',
    entry_point='pdomains.block_picking:BlockEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-block-picking-v1',
    entry_point='pdomains.block_picking_dev:BlockEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-heaven-hell-v0',
    entry_point='pdomains.heaven_hell:HeavenHellEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-ordered-tool-delivery-v0',
    entry_point='pdomains.ordered_tool_delivery:OrderedToolDeliveryEnv',
    max_episode_steps=20,
)

register(
    id='pdomains-speedy-tool-delivery-v0',
    entry_point='pdomains.speedy_tool_delivery:SpeedyToolDeliveryEnv',
    max_episode_steps=20,
)