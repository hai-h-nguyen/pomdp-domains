import numpy as np

class EpisodeLogger:
    def __init__(self, filename):
        self.filename = filename

    def log_episode(self, episode):
        valid_episode = []
        for i in range(len(episode)):
            obs, action, next_obs, reward, done, true_step_cnt = episode[i]
            if np.linalg.norm(action) > 0:
                valid_episode.append((obs, action, next_obs, reward, done))
        if valid_episode:
            assert len(valid_episode) == true_step_cnt
            obs, action, next_obs, reward, done = zip(*valid_episode)
            np.savez(self.filename, obs=np.array(obs), action=np.array(action), 
                     next_obs=np.array(next_obs), reward=np.array(reward), done=np.array(done))

# Example usage
logger = EpisodeLogger('episode_logs.npz')

# Example episode data
episode = [
    (np.array([1, 2, 3]), np.array([0.1, 0.2]), np.array([4, 5, 6]), 1, False, 1),
    (np.array([4, 5, 6]), np.array([0.0, 0.0]), np.array([4, 5, 6]), 0, False, 1),  # Action norm is 0, next_obs is same as obs
    (np.array([4, 5, 6]), np.array([0.0, 0.0]), np.array([4, 5, 6]), 0, False, 1),  # Action norm is 0, next_obs is same as obs
    (np.array([4, 5, 6]), np.array([0.0, 0.0]), np.array([4, 5, 6]), 0, False, 1),  # Action norm is 0, next_obs is same as obs
    (np.array([4, 5, 6]), np.array([0.0, 0.0]), np.array([4, 5, 6]), 0, False, 1),  # Action norm is 0, next_obs is same as obs
    (np.array([4, 5, 6]), np.array([0.3, 0.4]), np.array([7, 8, 9]), 1, False, 2),
    (np.array([7, 8, 9]), np.array([0.0, 0.0]), np.array([7, 8, 9]), 1, True, 2)  # Action norm is 0, next_obs is same as obs
]

# Log the episode (skipping timesteps with zero actions)
logger.log_episode(episode)


# Log the episode (skipping timesteps with zero actions)
logger.log_episode(episode)
