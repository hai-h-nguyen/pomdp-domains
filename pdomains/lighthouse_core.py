# -*- coding: utf-8 -*-

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym_minigrid import minigrid

import curses
import time
from typing import Optional, Tuple, Any, Union, cast

import abc
import string
from typing import List, Dict, Any, Optional, Tuple, Union, Sequence, cast

import gym
import numpy as np
from gym.utils import seeding

import sys
sys.path.append('/home/hainh22/Github/allenact')

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.task import Task, TaskSampler
from allenact.utils.experiment_utils import set_seed
from allenact.utils.system import get_logger
from allenact_plugins.lighthouse_plugin.lighthouse_environment import (
    LightHouseEnvironment,
)
from allenact_plugins.lighthouse_plugin.lighthouse_sensors import get_corner_observation

DISCOUNT_FACTOR = 0.99
STEP_PENALTY = -0.01
FOUND_TARGET_REWARD = 1.0


class LightHouseTask(Task[LightHouseEnvironment], abc.ABC):
    """Defines an abstract embodied task in the light house gridworld.

    # Attributes

    env : The light house environment.
    sensor_suite: Collection of sensors formed from the `sensors` argument in the initializer.
    task_info : Dictionary of (k, v) pairs defining task goals and other task information.
    max_steps : The maximum number of steps an agent can take an in the task before it is considered failed.
    observation_space: The observation space returned on each step from the sensors.
    """

    def __init__(
        self,
        env: LightHouseEnvironment,
        sensors: Union[SensorSuite, List[Sensor]],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs,
    ) -> None:
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )

        self._last_action: Optional[int] = None

    @property
    def last_action(self) -> int:
        return self._last_action

    @last_action.setter
    def last_action(self, value: int):
        self._last_action = value

    def step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        self.last_action = action
        return super(LightHouseTask, self).step(action=action)

    def render(self, mode: str = "array", *args, **kwargs) -> np.ndarray:
        if mode == "array":
            return self.env.render(mode, **kwargs)
        elif mode in ["rgb", "rgb_array", "human"]:
            arr = self.env.render("array", **kwargs)
            colors = np.array(
                [
                    (31, 119, 180),
                    (255, 127, 14),
                    (44, 160, 44),
                    (214, 39, 40),
                    (148, 103, 189),
                    (140, 86, 75),
                    (227, 119, 194),
                    (127, 127, 127),
                    (188, 189, 34),
                    (23, 190, 207),
                ],
                dtype=np.uint8,
            )
            return colors[arr]
        else:
            raise NotImplementedError("Render mode '{}' is not supported.".format(mode))

class FindGoalLightHouseTask(LightHouseTask):
    _CACHED_ACTION_NAMES: Dict[int, Tuple[str, ...]] = {}

    def __init__(
        self,
        env: LightHouseEnvironment,
        sensors: Union[SensorSuite, List[Sensor]],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs,
    ):
        super().__init__(env, sensors, task_info, max_steps, **kwargs)

        self._found_target = False

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(2 * self.env.world_dim)

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        self.env.step(action)
        reward = STEP_PENALTY

        if np.all(self.env.current_position == self.env.goal_position):
            self._found_target = True
            reward += FOUND_TARGET_REWARD
        elif self.num_steps_taken() == self.max_steps - 1:
            reward = STEP_PENALTY / (1 - DISCOUNT_FACTOR)

        return RLStepResult(
            observation=self.get_observations(),
            reward=reward,
            done=self.is_done(),
            info=None,
        )

    def reached_terminal_state(self) -> bool:
        return self._found_target

    @classmethod
    def class_action_names(cls, world_dim: int = 2, **kwargs) -> Tuple[str, ...]:
        assert 1 <= world_dim <= 26, "Too many dimensions."
        if world_dim not in cls._CACHED_ACTION_NAMES:
            action_names = [
                "{}(+1)".format(string.ascii_lowercase[i] for i in range(world_dim))
            ]
            action_names.extend(
                "{}(-1)".format(string.ascii_lowercase[i] for i in range(world_dim))
            )
            cls._CACHED_ACTION_NAMES[world_dim] = tuple(action_names)

        return cls._CACHED_ACTION_NAMES[world_dim]

    def action_names(self) -> Tuple[str, ...]:
        return self.class_action_names(world_dim=self.env.world_dim)

    def close(self) -> None:
        pass

    def query_expert(
        self,
        expert_view_radius: int,
        return_policy: bool = False,
        deterministic: bool = False,
        **kwargs,
    ) -> Tuple[Any, bool]:
        view_tuple = get_corner_observation(
            env=self.env, view_radius=expert_view_radius, view_corner_offsets=None,
        )

        goal = self.env.GOAL
        wrong = self.env.WRONG_CORNER

        if self.env.world_dim == 1:
            left_view, right_view, hitting, last_action = view_tuple

            left = 1
            right = 0

            expert_action: Optional[int] = None
            policy: Optional[np.ndarray] = None

            if left_view == goal:
                expert_action = left
            elif right_view == goal:
                expert_action = right
            elif hitting != 2 * self.env.world_dim:
                expert_action = left if last_action == right else right
            elif left_view == wrong:
                expert_action = right
            elif right_view == wrong:
                expert_action = left
            elif last_action == 2 * self.env.world_dim:
                policy = np.array([0.5, 0.5])
            else:
                expert_action = last_action

            if policy is None:
                policy = np.array([expert_action == right, expert_action == left])

        elif self.env.world_dim == 2:

            tl, tr, bl, br, hitting, last_action = view_tuple

            wall = self.env.WALL

            d, r, u, l, none = 0, 1, 2, 3, 4

            if tr == goal:
                if hitting != r:
                    expert_action = r
                else:
                    expert_action = u
            elif br == goal:
                if hitting != d:
                    expert_action = d
                else:
                    expert_action = r
            elif bl == goal:
                if hitting != l:
                    expert_action = l
                else:
                    expert_action = d
            elif tl == goal:
                if hitting != u:
                    expert_action = u
                else:
                    expert_action = l

            elif tr == wrong and not any(x == wrong for x in [br, bl, tl]):
                expert_action = l
            elif br == wrong and not any(x == wrong for x in [bl, tl, tr]):
                expert_action = u
            elif bl == wrong and not any(x == wrong for x in [tl, tr, br]):
                expert_action = r
            elif tl == wrong and not any(x == wrong for x in [tr, br, bl]):
                expert_action = d

            elif all(x == wrong for x in [tr, br]) and not any(
                x == wrong for x in [bl, tl]
            ):
                expert_action = l
            elif all(x == wrong for x in [br, bl]) and not any(
                x == wrong for x in [tl, tr]
            ):
                expert_action = u

            elif all(x == wrong for x in [bl, tl]) and not any(
                x == wrong for x in [tr, br]
            ):
                expert_action = r
            elif all(x == wrong for x in [tl, tr]) and not any(
                x == wrong for x in [br, bl]
            ):
                expert_action = d

            elif hitting != none and tr == br == bl == tl:
                # Only possible if in 0 vis setting
                if tr == self.env.WRONG_CORNER or last_action == hitting:
                    if last_action == r:
                        expert_action = u
                    elif last_action == u:
                        expert_action = l
                    elif last_action == l:
                        expert_action = d
                    elif last_action == d:
                        expert_action = r
                    else:
                        raise NotImplementedError()
                else:
                    expert_action = last_action

            elif last_action == r and tr == wall:
                expert_action = u

            elif last_action == u and tl == wall:
                expert_action = l

            elif last_action == l and bl == wall:
                expert_action = d

            elif last_action == d and br == wall:
                expert_action = r

            elif last_action == none:
                expert_action = r

            else:
                expert_action = last_action

            policy = np.array(
                [
                    expert_action == d,
                    expert_action == r,
                    expert_action == u,
                    expert_action == l,
                ]
            )
        else:
            raise NotImplementedError("Can only query expert for world dims of 1 or 2.")

        if return_policy:
            return policy, True
        elif deterministic:
            return int(np.argmax(policy)), True
        else:
            return (
                int(np.argmax(np.random.multinomial(1, policy / (1.0 * policy.sum())))),
                True,
            )
