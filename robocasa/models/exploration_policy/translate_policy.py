import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from .base import ExplorationPolicy


class LRExplorationPolicy(ExplorationPolicy):
    """
    This exploration policy moves the robot left and right.
    """

    def __init__(self, env, action_scale=0.1):
        super().__init__(env)
        self.action_scale = action_scale

    def begin(self, velocity_mode="random", init_obs=None):
        super().begin(init_obs)
        # find the target first and second quat positions by rotating the robot init_quat by 45 degrees to the left and right
        self.target_pos_1 = self.init_robot_base_pos - np.array([0.8, 0.0, 0.0])
        self.target_pos_2 = self.init_robot_base_pos + np.array([0.8, 0.0, 0.0])
        self.target_pos_3 = self.init_robot_base_pos
        self.stage_one_done = False
        self.stage_two_done = False
        self.stage_three_done = False
        if velocity_mode == "random":
            self.velocity = np.random.uniform(0.5, 0.6)
        elif velocity_mode == "fixed":
            self.velocity = 0.55
        else:
            raise ValueError(f"Invalid velocity mode: {velocity_mode}")

        # Parameters for velocity scaling
        self.slow_distance = 0.05  # Distance threshold to start slowing down
        self.min_velocity = 0.1  # Minimum velocity when very close

    def _get_scaled_velocity(self, target_pos, current_pos, base_velocity):
        """
        Scale velocity based on distance to target.
        Returns slower velocity when very close to destination.
        """
        distance = np.abs(target_pos[0] - current_pos[0])
        print(
            f"Target pos: {target_pos}",
            f"Current pos: {current_pos}",
            f"Distance: {distance}",
        )

        if distance <= self.slow_distance:
            scale_factor = max(
                distance / self.slow_distance, 0.1
            )  # Minimum scale of 0.1
            scaled_velocity = (
                self.min_velocity + (base_velocity - self.min_velocity) * scale_factor
            )
            print(f"Scaled velocity: {scaled_velocity}")
        else:
            scaled_velocity = base_velocity

        return scaled_velocity

    def explore_env(self, obs):
        """
        Rotate the robot by 45 degrees to the left and then right and then back to the original position.
        """
        if not self.stage_one_done:
            stage_one_done = (
                np.abs(obs["robot0_base_pos"][0] - self.target_pos_1[0]) < 0.02
            )
            self.stage_one_done = stage_one_done
        elif not self.stage_two_done:
            stage_two_done = (
                np.abs(obs["robot0_base_pos"][0] - self.target_pos_2[0]) < 0.02
            )
            self.stage_two_done = stage_two_done
        elif not self.stage_three_done:
            stage_three_done = (
                np.abs(obs["robot0_base_pos"][0] - self.target_pos_3[0]) < 0.02
            )
            self.stage_three_done = stage_three_done
        if self.stage_one_done and self.stage_two_done and self.stage_three_done:
            return None
        if not self.stage_one_done:
            scaled_velocity = self._get_scaled_velocity(
                self.target_pos_1, obs["robot0_base_pos"], self.velocity
            )
            base_action = np.array([0.0, scaled_velocity, 0.0])
        elif not self.stage_two_done:
            scaled_velocity = self._get_scaled_velocity(
                self.target_pos_2, obs["robot0_base_pos"], self.velocity
            )
            base_action = np.array([0.0, -scaled_velocity, 0.0])
        elif not self.stage_three_done:
            scaled_velocity = self._get_scaled_velocity(
                self.target_pos_3, obs["robot0_base_pos"], self.velocity
            )
            base_action = np.array([0.0, scaled_velocity, 0.0])
        base_dict = {"base": base_action, "base_mode": np.array([1.0])}
        if hasattr(self.env, "robots"):
            base_vector = self.env.robots[0].create_action_vector(base_dict)
        else:
            base_vector = self.env.env.robots[0].create_action_vector(base_dict)
        return base_vector
