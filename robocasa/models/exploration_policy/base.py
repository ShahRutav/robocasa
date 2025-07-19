import numpy as np
from termcolor import colored

import robosuite
import robosuite.utils.transform_utils as T


class ExplorationPolicy:
    """
    Base class for exploration policies.
    Inherit from this class and implement the explore_env method.
    """

    def __init__(self, env):
        """
        Initialize the exploration policy.

        Args:
            env: The environment to explore
        """
        self.env = env
        self.initial_obs = None
        self.step_count = 0

    def begin(self, init_obs=None):
        """
        Begin exploration by storing the initial observation.
        Should be called before starting exploration.
        """
        if init_obs is None:
            self.initial_obs = self.env._get_observations(force_update=True)
        else:
            self.initial_obs = init_obs
        self.init_robot_base_pos = self.initial_obs["robot0_base_pos"]
        self.init_robot_base_quat = self.initial_obs["robot0_base_quat"]
        self.step_count = 0
        self.last_action = None
        print(colored("Exploration started. Initial observation stored.", "green"))

    def explore_env(self, obs):
        """
        Implement your exploration strategy here.

        Args:
            obs: Current observation from the environment

        Returns:
            action: Action to take in the environment
        """
        raise NotImplementedError("Subclasses must implement explore_env method")

    def get_last_action(self, *args, **kwargs): # the subclass can override this method to return the last action
        """
        Get the last action taken by the exploration policy.
        """
        return self.last_action.copy()

    def get_action(self, obs):
        """
        Get action from the exploration policy.

        Args:
            obs: Current observation from the environment

        Returns:
            action: Action to take in the environment
        """
        self.step_count += 1
        self.last_action = self.explore_env(obs)
        return self.last_action.copy() if self.last_action is not None else None

    def _check_if_quat_reached(self, target_quat, current_quat, threshold=1e-2):
        """
        Check if the current quaternion is close to the target quaternion.
        """
        quat_diff = T.quat_multiply(T.quat_inverse(target_quat), current_quat)
        euler_diff = T.mat2euler(T.quat2mat(quat_diff))
        condition = np.linalg.norm(euler_diff) < threshold
        return condition
