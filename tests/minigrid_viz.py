from gym_minigrid import minigrid
import numpy as np
import time


height = width = 4
grid = minigrid.Grid(width, height)

# # Generate the surrounding walls
# grid.horz_wall(0, 0)
# grid.horz_wall(0, height - 1)
# grid.vert_wall(0, 0)
# grid.vert_wall(width - 1, 0)

# # Place fake agent at the center
# agent_pos = np.array(self.positions[-1]) + 1 + self.world_length
# # grid.set(*agent_pos, None)
# agent = minigrid.Goal()
# agent.color = "red"
# grid.set(agent_pos[0], agent_pos[1], agent)
# agent.init_pos = tuple(agent_pos)
# agent.cur_pos = tuple(agent_pos)

# goal_pos = self.goal_position + self.world_length

# goal = minigrid.Goal()
# grid.set(goal_pos[0], goal_pos[1], goal)
# goal.init_pos = tuple(goal_pos)
# goal.cur_pos = tuple(goal_pos)

highlight_mask = np.zeros((height, width), dtype=bool)

# minx, maxx = max(1, agent_pos[0] - 5), min(height - 1, agent_pos[0] + 5)
# miny, maxy = max(1, agent_pos[1] - 5), min(height - 1, agent_pos[1] + 5)
# highlight_mask[minx : (maxx + 1), miny : (maxy + 1)] = True

img = grid.render(
    minigrid.TILE_PIXELS, 2, None, highlight_mask=highlight_mask
)

print(img)