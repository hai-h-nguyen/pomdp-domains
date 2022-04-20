from pdomains import *
import pytest
import matplotlib.pyplot as plt
import gym
import numpy as np

STEP_PENALTY = -0.01
FOUND_TARGET_REWARD = 1.0
DISCOUNT_FACTOR = 0.99


def setup_domain(world_length, view_radius, max_step):
    """creates a env member"""
    domain = gym.make('pdomains-light-house-1D-v0', world_length=world_length, view_radius=view_radius, max_step=max_step)
    domain.reset()
    return domain


def test_reset():
    env = setup_domain(11, 1, 20)
    env.reset()
    assert(env.goal_position in [0, 10])
    assert(env.wall_position + env.goal_position == 10)
    assert(env.current_position in [5])

    env = setup_domain(11, 1, 20)
    env.reset()
    assert(env.goal_position in [0, 10])
    assert(env.wall_position + env.goal_position == 10)
    assert(env.current_position in [5])

    env = setup_domain(11, 1, 20)
    env.reset()
    assert(env.goal_position in [0, 10])
    assert(env.wall_position + env.goal_position == 10)
    assert(env.current_position in [5])

    env = setup_domain(11, 1, 20)
    env.reset()
    assert(env.goal_position in [0, 10])
    assert(env.wall_position + env.goal_position == 10)
    assert(env.current_position in [5])

    left = 0
    right = 0

    for _ in range(100):
        env = setup_domain(21, 3, 100)
        env.reset()

        if env.goal_position in [0]:
            left += 1

        if env.goal_position in [20]:
            right += 1

    assert (abs(left - right) < 30)


def test_observation():
    env = setup_domain(11, 1, 20)

    obs = env.reset(goal_position=0)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    obs, reward, _, _ = env.step(0)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(0)
    np.testing.assert_array_equal(obs, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(0)
    np.testing.assert_array_equal(obs, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(0)
    np.testing.assert_array_equal(obs, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    assert (reward == STEP_PENALTY)

    obs, reward, done, _ = env.step(0)
    np.testing.assert_array_equal(obs, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    assert (reward == STEP_PENALTY + FOUND_TARGET_REWARD)
    assert (done is True)

    env = setup_domain(11, 2, 20)

    obs = env.reset(goal_position=0)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])    

    obs, reward, _, _ = env.step(1)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])    
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(1)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])    
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(0)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])    
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(1)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])    
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(1)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1])    
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(1)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0])    
    assert (reward == STEP_PENALTY)

    # Reach the wall
    obs, reward, _, _ = env.step(1)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0])    
    assert (reward == STEP_PENALTY)

    # Go beyond the wall (stay the same)
    obs, reward, _, _ = env.step(1)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0])    
    assert (reward == STEP_PENALTY)

    # Go back
    obs, reward, _, _ = env.step(0)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0])    
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(0)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1])    
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(0)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])    
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(0)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])    
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(0)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])    
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(0)
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])    
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(0)
    np.testing.assert_array_equal(obs, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])    
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(0)
    np.testing.assert_array_equal(obs, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])    
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(0)
    np.testing.assert_array_equal(obs, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])    
    assert (reward == STEP_PENALTY)

    obs, reward, _, _ = env.step(0)
    np.testing.assert_array_equal(obs, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])    
    assert (reward == STEP_PENALTY + FOUND_TARGET_REWARD)


def test_timeout():
    env = setup_domain(15, 2, 20)
    env.reset(goal_position=0)

    for i in range(20):
        _, reward, done, _ = env.step(1)

        if i < 19:
            assert (done is False)

    assert (reward == STEP_PENALTY/(1 - DISCOUNT_FACTOR))
    assert(done is True)

    env = setup_domain(15, 2, 25)
    env.reset(goal_position=14)

    for i in range(25):
        _, reward, done, _ = env.step(0)

        if i < 24:
            assert (done is False)

    assert (reward == STEP_PENALTY/(1 - DISCOUNT_FACTOR))
    assert(done is True)


if __name__ == "__main__":
    pytest.main([__file__])
