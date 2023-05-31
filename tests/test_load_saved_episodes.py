import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max", type=int, default=1, help="Maximum index of file")

args = parser.parse_args()

for i in range(args.max):
    # Load the npz file
    data = np.load(f'{i}.npz')

    # Print the content
    # print('Observation:', data['obs'])
    # print('Action:', data['action'])
    # print('Next Observation:', data['next_obs'])
    # print('Reward:', data['reward'])
    # print('Done:', data['done'])

    assert len(data['obs']) == len(data['action']) == len(data['next_obs']) == len(data['reward']) == len(data['done']), i
    assert (data['obs'][1:] == data['next_obs'][:-1]).all()
    assert data['reward'][-1] > 0, i

    print(f"Episode {i} length {len(data['obs'])}")