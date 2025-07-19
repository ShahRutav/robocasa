import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from .base import ExplorationPolicy


class RotateExplorationPolicy(ExplorationPolicy):
    """
    This exploration policy rotates the robot by 45 degrees to the left and then right and then back to the original position.
    """

    def __init__(self, env, action_scale=0.1):
        super().__init__(env)
        self.action_scale = action_scale

    def begin(self, velocity_mode="random", init_obs=None):
        super().begin(init_obs)
        # find the target first and second quat positions by rotating the robot init_quat by 45 degrees to the left and right
        self.target_quat_1 = T.quat_multiply(
            self.init_robot_base_quat,
            T.mat2quat(T.euler2mat(np.array([0, 0, np.pi / 4]))),
        )
        self.target_quat_2 = T.quat_multiply(
            self.init_robot_base_quat,
            T.mat2quat(T.euler2mat(np.array([0, 0, -np.pi / 4]))),
        )
        self.target_quat_3 = self.init_robot_base_quat
        self.stage_one_done = False
        self.stage_two_done = False
        self.stage_three_done = False
        if velocity_mode == "random":
            self.velocity = np.random.uniform(0.2, 0.3)
        elif velocity_mode == "fixed":
            self.velocity = 0.25
        else:
            raise ValueError(f"Invalid velocity mode: {velocity_mode}")

    def explore_env(self, obs):
        """
        Rotate the robot by 45 degrees to the left and then right and then back to the original position.
        """
        if not self.stage_one_done:
            stage_one_done = self._check_if_quat_reached(
                self.target_quat_1, obs["robot0_base_quat"]
            )
            self.stage_one_done = stage_one_done
        elif not self.stage_two_done:
            stage_two_done = self._check_if_quat_reached(
                self.target_quat_2, obs["robot0_base_quat"]
            )
            self.stage_two_done = stage_two_done
        elif not self.stage_three_done:
            stage_three_done = self._check_if_quat_reached(
                self.target_quat_3, obs["robot0_base_quat"]
            )
            self.stage_three_done = stage_three_done
        if self.stage_one_done and self.stage_two_done and self.stage_three_done:
            return None
        if not self.stage_one_done:
            base_action = np.array([0.0, 0.0, self.velocity])
        elif not self.stage_two_done:
            base_action = np.array([0.0, 0.0, -self.velocity])
        elif not self.stage_three_done:
            base_action = np.array([0.0, 0.0, self.velocity])
        base_dict = {"base": base_action, "base_mode": np.array([1.0])}
        if hasattr(self.env, "robots"):
            base_vector = self.env.robots[0].create_action_vector(base_dict)
        else:
            base_vector = self.env.env.robots[0].create_action_vector(base_dict)
        return base_vector
