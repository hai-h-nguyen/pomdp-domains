import numpy as np


class EpisodeLogger:
    """
    Simple class to save data for expert demonstrations
    """
    def __init__(self, filename):
        self.filename = filename

    def log_episode(self, episode, save_state):
        valid_episode = []
        for i in range(len(episode)):
            obs, action, next_obs, reward, done, true_step_cnt, \
            state, next_state = episode[i]
            valid_episode.append((obs, action, next_obs, reward, done, state, next_state))

        if valid_episode:
            assert len(valid_episode) == true_step_cnt, f"{len(valid_episode)} != {true_step_cnt}"

            obs, action, next_obs, reward, done, state, next_state = zip(*valid_episode)

            np.savez(self.filename, obs=np.array(obs), action=np.array(action),
                     next_obs=np.array(next_obs), reward=np.array(reward), done=np.array(done))

            if save_state:
                np.savez('state-' + self.filename, obs=np.array(state), action=np.array(action),
                        next_obs=np.array(next_state), reward=np.array(reward), done=np.array(done))


def map_angles(angles):
    mapped_angles = []
    for angle in angles:
        mapped_angles.extend([np.sin(angle), np.cos(angle)])
    return mapped_angles
