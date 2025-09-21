"""

For demo robots, the robot0_right_gripper_mount is eef link and mobilebase0_center is the base link
For GR1 robots, the p
mobilebase0_center
"""
import os
import h5py
import json
import cv2
import argparse
import numpy as np
import pickle
from termcolor import colored

import robosuite
import robocasa
from robosuite.utils.binding_utils import MjSimState
from robosuite.utils import transform_utils as T
from robosuite.controllers import (
    load_composite_controller_config,
    load_part_controller_config,
)

from icrt.util.casa_utils import make_env, reset_to
import random
import torch


def control_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(
        "Simple demonstration replay using casa_utils.make_env and reset_to"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path to an HDF5 file containing demonstration episodes.",
    )
    parser.add_argument(
        "--robots",
        type=str,
        default="DemoTwoHand",
        help="Name of the robot(s). E.g. 'DemoTwoFingered', 'DemoTwoHand', etc.",
    )
    parser.add_argument(
        "--robot_names",
        type=str,
        default=["PandaOmron", "DemoTwoHand", "GR1TwoHand"],
        nargs="+",
        help="Name of the robot(s). E.g. 'DemoTwoFingered', 'DemoTwoHand', etc.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="(Optional) Controller name, e.g. 'WHOLE_BODY_MINK_IK', 'OSC_POSE'",
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=5,
        help="Number of episodes to replay from the HDF5 file.",
    )
    parser.add_argument(
        "--render", action="store_true", help="If set, enable on-screen rendering."
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="If set, record and save replay videos.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="temp",
        help="Directory to save videos if --save_video is used.",
    )
    parser.add_argument(
        "--video_prefix",
        type=str,
        default="rollout",
        help="Prefix for saved videos if --save_video is used.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for consistency (optional)."
    )
    parser.add_argument("-cf", "--control_freq", type=int, default=20)
    args = parser.parse_args()

    # Set random seeds if you want reproducibility
    control_seed(args.seed)

    reachability_errors_per_robot = {}
    for robot_run in args.robot_names:
        reachability_errors_per_robot[robot_run] = []
        print("Running robot: ", robot_run)
        args.robots = robot_run
        # (Optional) if you have a Mink IK or other JSON config, load it:
        args.controller = (
            "WHOLE_BODY_MINK_IK" if "GR1" in args.robots else args.controller
        )
        if "PandaOmron" in args.robots:
            args.controller = "OSC_POSE"

        # 1) Create the environment via casa_utils.make_env
        #    If your make_env requires specific arguments, adapt as needed.
        env, env_kwargs = make_env(
            args.prompt_file,  # or you can pass None if your make_env doesn't need it
            keys_info={"image_keys": []},
            args=args,
        )

        # 2) Open the HDF5 file
        if not os.path.exists(args.prompt_file):
            raise FileNotFoundError(f"HDF5 file does not exist: {args.prompt_file}")
        with h5py.File(args.prompt_file, "r") as f:
            data_root = f["data"] if "data" in f else f
            ep_names = sorted(list(data_root.keys()))
            num_eps = len(ep_names)
            print(colored(f"Found {num_eps} episodes in {args.prompt_file}", "green"))

            n_eval = min(args.n_eval, num_eps)
            successes = 0

            # 3) For each episode, reset to initial state and replay the stored actions
            for i in range(n_eval):
                ep_name = ep_names[i]
                ep_group = data_root[ep_name]
                print(colored(f"[Eval {i+1}/{n_eval}] Episode: {ep_name}", "blue"))

                states = ep_group["states"][:]  # shape: (T, ?)
                actions = ep_group["actions"][:]  # shape: (T, action_dim)

                # If there's extra metadata, you can read it:
                ep_meta = ep_group.attrs.get("ep_meta", {})
                if isinstance(ep_meta, str):
                    ep_meta = json.loads(ep_meta)

                env.set_ep_meta(ep_meta)
                env.reset()

                success = False
                frames = []

                obs = env._get_observations()
                for t in range(len(actions)):
                    action = actions[t]
                    obs, rew, done, info = env.step(action)
                    if env._check_success():
                        success = True
                        break
                    if args.render:
                        env.render()

                    # If saving video, record frames from, e.g., "robot0_agentview_sideleft"
                    if args.save_video:
                        frame = env.sim.render(
                            camera_name="robot0_agentview_sideleft",
                            width=128,
                            height=128,
                        )
                        # Convert from RGB to BGR if you want OpenCV convention
                        frame = frame[..., ::-1]
                        frames.append(frame)

                if success:
                    successes += 1
                    print(colored("SUCCESS", "green"))
                else:
                    print(colored("FAIL", "red"))

                # 5) Save video if requested
                if args.save_video and len(frames) > 0:
                    vid_name = f"{args.video_prefix}_{ep_name}.mp4"
                    vid_name = os.path.join(args.video_dir, vid_name)
                    os.makedirs(args.video_dir, exist_ok=True)
                    h, w, _ = frames[0].shape
                    out = cv2.VideoWriter(
                        vid_name,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        20,  # FPS
                        (w, h),
                    )
                    for frame in frames:
                        out.write(frame)
                    out.release()
                    print(colored(f"Saved video: {vid_name}", "yellow"))

        # 6) Print final success rate (no CSV saving)
        rate = successes / n_eval
        print(
            colored(f"Final success rate = {successes}/{n_eval} = {rate:.2f}", "green")
        )
        env.close()


if __name__ == "__main__":
    main()
