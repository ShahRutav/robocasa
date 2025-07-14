"""
A script to explore the environment using an exploration policy.
Similar to collect_demos.py but uses automated exploration instead of human input.
"""

import argparse
import datetime
import json
import os
import time
import random
from tkinter import W
import numpy as np
from termcolor import colored

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

import robocasa
import robocasa.macros as macros
from robocasa.models.exploration_policy import RotateExplorationPolicy


def control_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def explore_environment(
    env,
    policy,
    max_steps=1000,
    render=True,
    max_fr=None,
    print_info=True,
    env_name=None,
):
    """
    Explore the environment using the provided exploration policy.

    Args:
        env: Environment to explore
        policy: ExplorationPolicy instance
        max_steps: Maximum number of steps to explore
        render: Whether to render the environment
        max_fr: Maximum frame rate
        print_info: Whether to print exploration info
        env_name: Name of the environment

    Returns:
        success: Whether exploration completed successfully
    """

    # Get episode metadata
    ep_meta = None
    if hasattr(env, "meta_file_path"):
        ep_meta_file_name = env.meta_file_path()
        if os.path.exists(ep_meta_file_name):
            ep_meta = json.load(open(ep_meta_file_name, "r"))
            env.set_ep_meta(ep_meta)

    env.reset()

    if ep_meta is None:
        ep_meta = env.get_ep_meta()

    if print_info:
        lang = ep_meta.get("lang", None)
        if lang:
            print(colored(f"Task: {lang}", "green"))

    if render:
        env.render()

    # Initialize exploration
    policy.begin()
    obs = env._get_observations()

    # Exploration loop
    for step in range(max_steps):
        start = time.time()

        # Get action from exploration policy
        action = policy.get_action(obs)
        if action is None:
            print(colored("Task completed successfully!", "green"))
            break

        # Take step in environment
        obs, reward, done, info = env.step(action)

        if render:
            env.render()

        # Print progress
        if step % 100 == 0:
            print(f"Exploration step {step}/{max_steps}")

        # Check for task completion
        if env._check_success():
            print(colored("Task completed successfully!", "green"))
            break

        # Limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    print(f"Exploration completed after {policy.step_count} steps")
    return True


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Kitchen")
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Maximum exploration steps"
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Demo",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be, eg. 'NONE' or 'WHOLE_BODY_IK', etc. Or path to controller json file",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Which camera to use for rendering",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--renderer", type=str, default="mjviewer", choices=["mjviewer", "mujoco"]
    )
    parser.add_argument(
        "--max_fr", default=30, type=int, help="If specified, limit the frame rate"
    )
    parser.add_argument("--layout", type=int, nargs="+", default=-1)
    parser.add_argument(
        "--style", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11]
    )
    parser.add_argument("--generative_textures", action="store_true")
    parser.add_argument("--ref_frame", type=str, default="world")
    parser.add_argument(
        "--action_scale", type=float, default=0.1, help="Scale for random actions"
    )

    args = parser.parse_args()

    control_seed(args.seed)

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )

    controller_config["body_parts"]["right"]["input_type"] = "delta"
    controller_config["body_parts"]["right"]["input_ref_frame"] = args.ref_frame
    if "left" in controller_config["body_parts"]:
        controller_config["body_parts"]["left"]["input_type"] = "delta"
        controller_config["body_parts"]["left"]["input_ref_frame"] = args.ref_frame

    env_name = args.environment

    # Create argument configuration
    config = {
        "env_name": env_name,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    if args.generative_textures is True:
        config["generative_textures"] = "100p"

    # Check if we're using a multi-armed environment
    if "TwoArm" in env_name:
        config["env_configuration"] = args.config

    # Configure for kitchen environments
    if env_name in ["Lift"]:
        mirror_actions = False
        if args.camera is None:
            args.camera = "agentview"
        elif args.camera == "free":
            args.camera = None
    else:
        mirror_actions = False
        config["seed"] = args.seed
        config["layout_ids"] = args.layout
        config["style_ids"] = args.style
        if args.camera is None:
            args.camera = "robot0_frontview"
        elif args.camera == "free":
            args.camera = None
        config["translucent_robot"] = True
        config["obj_instance_split"] = "A"

    # Create environment
    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer=args.renderer,
    )

    # Wrap with visualization wrapper
    env = VisualizationWrapper(env)

    # Create exploration policy
    policy = RotateExplorationPolicy(env, action_scale=args.action_scale)

    # You can replace the above line with your own exploration policy:
    # policy = YourCustomExplorationPolicy(env)

    # Start exploration
    print(colored("Starting environment exploration...", "blue"))
    success = explore_environment(
        env=env,
        policy=policy,
        max_steps=args.max_steps,
        render=(args.renderer != "mjviewer"),
        max_fr=args.max_fr,
        env_name=config["env_name"],
    )

    env.close()
    print(colored("Exploration completed!", "blue"))
