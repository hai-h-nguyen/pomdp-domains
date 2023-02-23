This is the repo that store several pomdp domains that I use for my research.

### Setup
- Install `pip3 install -e .`
- You will need MuJoCo to run for all domains except for Car-Flag
- You will need to install https://github.com/hai-h-nguyen/pomdp_robot_domains to run Block-Picking

### Domains
- Car-Flag: A car must go to the green flag (which can be on either side with equal prob.). The car is given the direction of the green flag when it is close to the blue flag. It is given a penalty when it goes to the red flag.

<p align="center">
  <img src="./images/car-flag.png" height="200" />
</p>

- Ant-Tag: An ant must move close enough to an opponent which tends to move away from the ant. The ant does not usually observe the opponent's position except when the opponent is within the blue sphere.

<p align="center">
  <img src="./images/ant-tag.png" height="200" />
</p>

- Two-Boxes: Using the history of angles and positions, the finger must check the sizes of two boxes to see if they are equal. If yes, it goes to the right to finish the task, else go to the left.

<p align="center">
  <img src="./images/two-boxes.png" height="200" />
</p>

- Ant-Heaven-Hell: Similar to Car-Flag but in 2D, the ant must go to the green area whose side are given when it is within the blue sphere. It is given a penalty when it goes to the red area.

<p align="center">
  <img src="./images/ant-hh.png" height="200" />
</p>

- Block-Picking: The agent has to pick on block up from two blocks using colorless depth images. Only one block is pickable and the agent does not know which one. In the current version of the domain, the positions of the blocks are fixed but they switch their positions randomly. To test the domain, run tests/test_block_picking.py. This script will run an expert which know exactly which block to pick.

<p align="center">
  <img src="./images/block_picking.png" height="200" />
</p>

### Cite
If you use this repo in your published work, please cite as


```
@misc{nguyen2021penvs,
  author={Hai, Nguyen},
  title={Pomdp Robot Domains},
  year={2021},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/hai-h-nguyen/pomdp-domains}},
}
```
