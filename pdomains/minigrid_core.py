import random
from typing import Tuple, Any, List, Dict, Optional, Union, Callable, Sequence, cast
import sys

import gym
import networkx as nx
import numpy as np
from gym.utils import seeding
from gym_minigrid.envs import CrossingEnv
from gym_minigrid.minigrid import (
    DIR_TO_VEC,
    IDX_TO_OBJECT,
    MiniGridEnv,
    OBJECT_TO_IDX,
)

sys.path.append('/home/hainh22/Github/allenact')

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.task import Task, TaskSampler
from allenact.utils.system import get_logger
from allenact_plugins.minigrid_plugin.minigrid_environments import (
    AskForHelpSimpleCrossing,
)


class MiniGridTask(Task[CrossingEnv]):
    _ACTION_NAMES: Tuple[str, ...] = ("left", "right", "forward")
    _ACTION_IND_TO_MINIGRID_IND = tuple(
        MiniGridEnv.Actions.__members__[name].value for name in _ACTION_NAMES
    )
    _CACHED_GRAPHS: Dict[str, nx.DiGraph] = {}
    _NEIGHBOR_OFFSETS = tuple(
        [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1),]
    )

    _XY_DIFF_TO_AGENT_DIR = {
        tuple(vec): dir_ind for dir_ind, vec in enumerate(DIR_TO_VEC)
    }

    """ Task around a MiniGrid Env, allows interfacing allenact with
    MiniGrid tasks. (currently focussed towards LavaCrossing)
    """

    def __init__(
        self,
        env: Union[CrossingEnv],
        sensors: Union[SensorSuite, List[Sensor]],
        task_info: Dict[str, Any],
        max_steps: int,
        task_cache_uid: Optional[str] = None,
        corrupt_expert_within_actions_of_goal: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self._graph: Optional[nx.DiGraph] = None
        self._minigrid_done = False
        self._task_cache_uid = task_cache_uid
        self.corrupt_expert_within_actions_of_goal = (
            corrupt_expert_within_actions_of_goal
        )
        self.closest_agent_has_been_to_goal: Optional[float] = None

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(len(self._ACTION_NAMES))

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        return self.env.render(mode=mode)

    def step(self, action: int) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        minigrid_obs, reward, self._minigrid_done, info = self.env.step(
            action=self._ACTION_IND_TO_MINIGRID_IND[action]
        )

        # self.env.render()

        return RLStepResult(
            observation=self.get_observations(minigrid_output_obs=minigrid_obs),
            reward=reward,
            done=self.is_done(),
            info=info,
        )

    def get_observations(
        self, *args, minigrid_output_obs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Any:
        return self.sensor_suite.get_observations(
            env=self.env, task=self, minigrid_output_obs=minigrid_output_obs
        )

    def reached_terminal_state(self) -> bool:
        return self._minigrid_done

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._ACTION_NAMES

    def close(self) -> None:
        pass

    @property
    def graph_created(self):
        return self._graph is not None

    @property
    def graph(self):
        if self._graph is None:
            if self._task_cache_uid is not None:
                if self._task_cache_uid not in self._CACHED_GRAPHS:
                    self._CACHED_GRAPHS[self._task_cache_uid] = self.generate_graph()
                self._graph = self._CACHED_GRAPHS[self._task_cache_uid]
            else:
                self._graph = self.generate_graph()
        return self._graph

    @graph.setter
    def graph(self, graph: nx.DiGraph):
        self._graph = graph

    @classmethod
    def possible_neighbor_offsets(cls) -> Tuple[Tuple[int, int, int], ...]:
        # Tuples of format:
        # (X translation, Y translation, rotation by 90 degrees)
        # A constant is returned, this function can be changed if anything
        # more complex needs to be done.

        # offsets_superset = itertools.product(
        #     [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]
        # )
        #
        # valid_offsets = []
        # for off in offsets_superset:
        #     if (int(off[0] != 0) + int(off[1] != 0) + int(off[2] != 0)) == 1:
        #         valid_offsets.append(off)
        #
        # return tuple(valid_offsets)

        return cls._NEIGHBOR_OFFSETS

    @classmethod
    def _add_from_to_edge(
        cls, g: nx.DiGraph, s: Tuple[int, int, int], t: Tuple[int, int, int],
    ):
        """Adds nodes and corresponding edges to existing nodes.
        This approach avoids adding the same edge multiple times.
        Pre-requisite knowledge about MiniGrid:
        DIR_TO_VEC = [
            # Pointing right (positive X)
            np.array((1, 0)),
            # Down (positive Y)
            np.array((0, 1)),
            # Pointing left (negative X)
            np.array((-1, 0)),
            # Up (negative Y)
            np.array((0, -1)),
        ]
        or
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }
        This also implies turning right (clockwise) means:
            agent_dir += 1
        """

        s_x, s_y, s_rot = s
        t_x, t_y, t_rot = t

        x_diff = t_x - s_x
        y_diff = t_y - s_y
        angle_diff = (t_rot - s_rot) % 4

        # If source and target differ by more than one action, continue
        if (x_diff != 0) + (y_diff != 0) + (angle_diff != 0) != 1 or angle_diff == 2:
            return

        action = None
        if angle_diff == 1:
            action = "right"
        elif angle_diff == 3:
            action = "left"
        elif cls._XY_DIFF_TO_AGENT_DIR[(x_diff, y_diff)] == s_rot:
            # if translation is the same direction as source
            # orientation, then it's a valid forward action
            action = "forward"
        else:
            # This is when the source and target aren't one action
            # apart, despite having dx=1 or dy=1
            pass

        if action is not None:
            g.add_edge(s, t, action=action)

    def _add_node_to_graph(
        self,
        graph: nx.DiGraph,
        s: Tuple[int, int, int],
        valid_node_types: Tuple[str, ...],
        attr_dict: Dict[Any, Any] = None,
        include_rotation_free_leaves: bool = False,
    ):
        if s in graph:
            return
        if attr_dict is None:
            get_logger().warning("adding a node with neighbor checks and no attributes")
        graph.add_node(s, **attr_dict)

        if include_rotation_free_leaves:
            rot_free_leaf = (*s[:-1], None)
            if rot_free_leaf not in graph:
                graph.add_node(rot_free_leaf)
            graph.add_edge(s, rot_free_leaf, action="NA")

        if attr_dict["type"] in valid_node_types:
            for o in self.possible_neighbor_offsets():
                t = (s[0] + o[0], s[1] + o[1], (s[2] + o[2]) % 4)
                if t in graph and graph.nodes[t]["type"] in valid_node_types:
                    self._add_from_to_edge(graph, s, t)
                    self._add_from_to_edge(graph, t, s)

    def generate_graph(self,) -> nx.DiGraph:
        """The generated graph is based on the fully observable grid (as the
        expert sees it all).

        env: environment to generate the graph over
        """

        image = self.env.grid.encode()
        width, height, _ = image.shape
        graph = nx.DiGraph()

        # In fully observable grid, there shouldn't be any "unseen"
        # Currently dealing with "empty", "wall", "goal", "lava"

        valid_object_ids = np.sort(
            [OBJECT_TO_IDX[o] for o in ["empty", "wall", "lava", "goal"]]
        )

        assert np.all(np.union1d(image[:, :, 0], valid_object_ids) == valid_object_ids)

        # Grid to nodes
        for x in range(width):
            for y in range(height):
                for rotation in range(4):
                    type, color, state = image[x, y]
                    self._add_node_to_graph(
                        graph,
                        (x, y, rotation),
                        attr_dict={
                            "type": IDX_TO_OBJECT[type],
                            "color": color,
                            "state": state,
                        },
                        valid_node_types=("empty", "goal"),
                    )
                    if IDX_TO_OBJECT[type] == "goal":
                        if not graph.has_node("unified_goal"):
                            graph.add_node("unified_goal")
                        graph.add_edge((x, y, rotation), "unified_goal")

        return graph

    def query_expert(self, **kwargs) -> Tuple[int, bool]:
        if self._minigrid_done:
            get_logger().warning("Episode is completed, but expert is still queried.")
            return -1, False

        paths = []
        agent_x, agent_y = self.env.agent_pos
        agent_rot = self.env.agent_dir
        source_state_key = (agent_x, agent_y, agent_rot)
        assert source_state_key in self.graph

        paths.append(nx.shortest_path(self.graph, source_state_key, "unified_goal"))

        if len(paths) == 0:
            return -1, False

        shortest_path_ind = int(np.argmin([len(p) for p in paths]))

        if self.closest_agent_has_been_to_goal is None:
            self.closest_agent_has_been_to_goal = len(paths[shortest_path_ind]) - 1
        else:
            self.closest_agent_has_been_to_goal = min(
                len(paths[shortest_path_ind]) - 1, self.closest_agent_has_been_to_goal
            )

        if (
            self.corrupt_expert_within_actions_of_goal is not None
            and self.corrupt_expert_within_actions_of_goal
            >= self.closest_agent_has_been_to_goal
        ):
            return (
                int(self.env.np_random.randint(0, len(self.class_action_names()))),
                True,
            )

        if len(paths[shortest_path_ind]) == 2:
            # Since "unified_goal" is 1 step away from actual goals
            # if a path like [actual_goal, unified_goal] exists, then
            # you are already at a goal.
            get_logger().warning(
                "Shortest path computations suggest we are at"
                " the target but episode does not think so."
            )
            return -1, False

        next_key_on_shortest_path = paths[shortest_path_ind][1]
        return (
            self.class_action_names().index(
                self.graph.get_edge_data(source_state_key, next_key_on_shortest_path)[
                    "action"
                ]
            ),
            True,
        )


class AskForHelpSimpleCrossingTask(MiniGridTask):
    _ACTION_NAMES = ("left", "right", "forward", "toggle")
    _ACTION_IND_TO_MINIGRID_IND = tuple(
        MiniGridEnv.Actions.__members__[name].value for name in _ACTION_NAMES
    )
    _CACHED_GRAPHS: Dict[str, nx.DiGraph] = {}

    def __init__(
        self,
        env: AskForHelpSimpleCrossing,
        sensors: Union[SensorSuite, List[Sensor]],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs,
    ):
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )

        self.did_toggle: List[bool] = []

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        self.did_toggle.append(self._ACTION_NAMES[action] == "toggle")
        return super(AskForHelpSimpleCrossingTask, self)._step(action=action)

    def metrics(self) -> Dict[str, Any]:
        return {
            **super(AskForHelpSimpleCrossingTask, self).metrics(),
            "toggle_percent": float(
                sum(self.did_toggle) / max(len(self.did_toggle), 1)
            ),
        }
