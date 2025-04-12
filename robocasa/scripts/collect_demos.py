"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import argparse
from copy import deepcopy
import datetime
import json
import os
import time
import pickle
from glob import glob

import h5py
import imageio
import mujoco
import torch
import random
import numpy as np
from termcolor import colored

# from robosuite import load_controller_config
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.control_utils import convert_delta_to_abs_action

import robocasa
import robocasa.macros as macros
from robocasa.models.fixtures import FixtureType
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format

from icrt.util.misc import confirm_user


def control_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def is_empty_input_spacemouse(action_dict):
    if ("left_delta" in action_dict.keys()) and ("right_delta" in action_dict.keys()):
        if (
            np.all(action_dict["left_delta"] == 0)
            and np.all(action_dict["right_delta"] == 0)
            and np.all(action_dict["base_mode"] == -1)
            and np.all(action_dict["base"] == 0)
        ):
            return True
        return False
    if (
        np.all(action_dict["right_delta"] == 0)
        and action_dict["base_mode"] == -1
        and np.all(action_dict["base"] == 0)
    ):
        return True

    return False


def collect_human_trajectory(
    env,
    device,
    arm,
    env_configuration,
    mirror_actions,
    render=True,
    max_fr=None,
    print_info=True,
    env_name=None,
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    ep_meta = None
    ep_meta_file_name = f"scene_configs/ep_meta_{env_name}.pkl"
    load_from_pickle = False
    if hasattr(env, "meta_file_path"):
        ep_meta_file_name = env.meta_file_path()
        # load the json file if it exists
        ep_meta = json.load(open(ep_meta_file_name, "r"))
        env.set_ep_meta(ep_meta)

    # if (load_from_pickle) and (env_name != "Lift"):
    #     ep_meta = pickle.load(open(ep_meta_file_name, "rb"))
    #     env.set_ep_meta(ep_meta)

    env.reset()

    # if (not load_from_pickle) and (env_name != "Lift"):
    #     ep_meta = env.get_ep_meta()
    #     pickle.dump(ep_meta, open(ep_meta_file_name, "wb"))
    print("ep_meta", ep_meta)
    # import ipdb; ipdb.set_trace()

    # print(json.dumps(ep_meta, indent=4))
    lang = None
    if ep_meta is None:
        ep_meta = env.get_ep_meta()
    if print_info:
        lang = ep_meta.get("lang", None)
        print(colored(f"Instruction: {lang}", "green"))

    # degugging: code block here to quickly test and close env
    # env.close()
    # return None, True

    if render:
        # ID = 2 always corresponds to agentview
        env.render()

    task_completion_hold_count = (
        -1
    )  # counter to collect 10 timesteps after reaching goal
    for d in device:
        d.start_control()

    nonzero_ac_seen = False

    # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    active_robot = env.robots[device[0].active_robot]
    zero_action = np.zeros(env.action_dim)
    zero_action_dict = {}
    arms = ["left", "right"] if args.dual_arm else ["left"]
    for arm in arms:
        # controller has absolute actions, so we need to set the initial action to be the current position
        zero_action = np.zeros(7)
        if active_robot.part_controllers[arm].input_type == "absolute":
            zero_action = convert_delta_to_abs_action(
                zero_action, active_robot, arm, env
            )
        zero_action_dict[f"{arm}"] = zero_action[: zero_action.shape[0] - 1]
        zero_action_dict[f"{arm}_gripper"] = zero_action[zero_action.shape[0] - 1 :]
    zero_action_dict["base_mode"] = -1
    zero_action_dict["base"] = np.zeros(3)
    zero_action = active_robot.create_action_vector(zero_action_dict)
    for _ in range(1):
        # do a dummy step thru base env to initalize things, but don't record the step
        if isinstance(env, DataCollectionWrapper):
            env.env.step(zero_action)
        else:
            env.step(zero_action)

    discard_traj = False

    success = False
    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Set active robot
        active_robot = env.robots[device[0].active_robot]
        input_ac_dict = {}
        for d_ind, d in enumerate(device):
            active_arm = d.active_arm
            # Get the newest action
            arm_ac_dict = d.input2action(mirror_actions=mirror_actions)
            if d_ind == 0:
                input_ac_dict = arm_ac_dict
            else:
                if arm_ac_dict is None:
                    continue
                for k, v in arm_ac_dict.items():
                    if active_arm in k:
                        input_ac_dict[k] = v

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            discard_traj = True
            break

        if "base_mode" not in input_ac_dict:
            input_ac_dict["base_mode"] = -1
            input_ac_dict["base"] = np.zeros(3)

        action_dict = deepcopy(input_ac_dict)

        # set arm actions
        for arm in active_robot.arms:
            controller_input_type = active_robot.part_controllers[arm].input_type
            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        # print(is_empty_input_spacemouse(action_dict))
        if is_empty_input_spacemouse(action_dict):
            if not nonzero_ac_seen:
                if render:
                    env.render()
                continue
        else:
            nonzero_ac_seen = True

        print("action_dict", action_dict)
        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [
            robot.create_action_vector(all_prev_gripper_actions[i])
            for i, robot in enumerate(env.robots)
        ]
        env_action[device[0].active_robot] = active_robot.create_action_vector(
            action_dict
        )
        env_action = np.concatenate(env_action)

        # Run environment step
        obs, _, _, _ = env.step(env_action)
        if render:
            env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            success = True
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

        # with open("/home/soroushn/tmp/model.xml", "w") as f:
        #     f.write(env.model.get_xml())
        # exit()

    if nonzero_ac_seen and hasattr(env, "ep_directory"):
        ep_directory = env.ep_directory
    else:
        ep_directory = None

    # cleanup for end of data collection episodes
    env.close()

    discard_traj = discard_traj or not success
    return ep_directory, discard_traj


def gather_demonstrations_as_hdf5(directory, out_dir, env_info, excluded_episodes=None):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.
    The strucure of the hdf5 file is as follows.
    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected
        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration
        demo2 (group)
        ...
    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    print("Saving hdf5 to", hdf5_path)
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print("Processing {} ...".format(ep_directory))
        if (excluded_episodes is not None) and (ep_directory in excluded_episodes):
            # print("\tExcluding this episode!")
            continue

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        obs = []
        actions_abs = []
        initial_obj_qpos = {}
        initial_qpos = None
        non_robot_qpos_idx = []
        index = 0
        # success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            obs.extend(dic["obs"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
                if "actions_abs" in ai:
                    actions_abs.append(ai["actions_abs"])
            if index == 0:
                initial_qpos = dic["initial_qpos"]
                initial_obj_qpos = dic["initial_obj_qpos"]
                non_robot_qpos_idx = dic["non_robot_qpos_idx"]
            # success = success or dic["successful"]

        if len(states) == 0:
            continue

        # # Add only the successful demonstration to dataset
        # if success:

        # print("Demonstration is successful and has been saved")
        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        del obs[-1]
        assert len(states) == len(actions)
        assert len(obs) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # store ep meta as an attribute
        ep_meta_path = os.path.join(directory, ep_directory, "ep_meta.json")
        if os.path.exists(ep_meta_path):
            with open(ep_meta_path, "r") as f:
                ep_meta = f.read()
            ep_data_grp.attrs["ep_meta"] = ep_meta
            ep_data_grp.attrs["non_robot_qpos_idx"] = non_robot_qpos_idx
            ep_data_grp.attrs["initial_qpos"] = initial_qpos
            print("non_robot_qpos_idx", non_robot_qpos_idx)

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        if len(actions_abs) > 0:
            print(np.array(actions_abs).shape)
            ep_data_grp.create_dataset("actions_abs", data=np.array(actions_abs))
        # write observations. Observations is dictionary of numpy arrays
        obs_grp = ep_data_grp.create_group("observations")
        for key, value in obs[0].items():
            obs_grp.create_dataset(key, data=np.array([o[key] for o in obs]))

        # initial object qpos is a dictionary
        ep_data_grp.create_group("initial_obj_qpos")
        for key, value in initial_obj_qpos.item().items():
            ep_data_grp["initial_obj_qpos"].create_dataset(key, data=value)

        # else:
        #     pass
        #     # print("Demonstration is unsuccessful and has NOT been saved")

    print("{} successful demos so far".format(num_eps))

    if num_eps == 0:
        f.close()
        return

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["robocasa_version"] = robocasa.__version__
    grp.attrs["robosuite_version"] = robosuite.__version__
    grp.attrs["mujoco_version"] = mujoco.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()

    return hdf5_path


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(robocasa.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="Kitchen")
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument(
        "--n_demos", type=int, default=50, help="Number of demonstrations to collect"
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
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--obj_groups",
        type=str,
        nargs="+",
        default=None,
        help="In kitchen environments, either the name of a group to sample object from or path to an .xml file",
    )

    parser.add_argument(
        "--camera",
        type=str,
        default="free",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be, eg. 'NONE' or 'WHOLE_BODY_IK', etc. Or path to controller json file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="spacemouse",
        choices=["keyboard", "keyboardmobile", "spacemouse", "dummy", "oculus"],
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dual_arm", action="store_true")
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
    args = parser.parse_args()

    control_seed(args.seed)
    for robot in args.robots:
        if "GR1" in robot:
            args.controller = "WHOLE_BODY_MINK_IK"
    print("Controller:", args.controller)
    # Get controller config
    # controller_config = load_controller_config(default_controller=args.controller)
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        # mink-speicific import. requires installing mink
        from robosuite.examples.third_party_controller.mink_controller import (
            WholeBodyMinkIK,
        )
    controller_config["body_parts"]["right"]["input_type"] = "absolute"
    controller_config["body_parts"]["right"]["input_ref_frame"] = args.ref_frame
    if "left" in controller_config["body_parts"]:
        controller_config["body_parts"]["left"]["input_type"] = "absolute"
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

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in env_name:
        config["env_configuration"] = args.config

    # Mirror actions if using a kitchen environment
    if env_name in ["Lift"]:  # add other non-kitchen tasks here
        if args.obj_groups is not None:
            print(
                "Specifying 'obj_groups' in non-kitchen environment does not have an effect."
            )
        mirror_actions = False
        if args.camera is None:
            args.camera = "agentview"
        # special logic: "free" camera corresponds to Null camera
        elif args.camera == "free":
            args.camera = None
    else:
        mirror_actions = False if args.device == "oculus" else True
        config["seed"] = args.seed
        config["layout_ids"] = args.layout
        config["style_ids"] = args.style
        ### update config for kitchen envs ###
        if args.obj_groups is not None:
            config.update({"obj_groups": args.obj_groups})
        if args.camera is None:
            args.camera = "robot0_frontview"
        # special logic: "free" camera corresponds to Null camera
        elif args.camera == "free":
            args.camera = None

        config["translucent_robot"] = True

        # by default use obj instance split A
        config["obj_instance_split"] = "A"
        # config["obj_instance_split"] = None
        # config["obj_registries"] = ("aigen",)

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

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime("%Y-%m-%d-%H-%M-%S")

    if not args.debug:
        # wrap the environment with data collection wrapper
        tmp_directory = "/tmp/{}".format(time_str)
        env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            # vendor_id=macros.SPACEMOUSE_VENDOR_ID,
            # product_id=macros.SPACEMOUSE_PRODUCT_ID,
        )
    elif args.device == "oculus":
        assert args.pos_sensitivity == 3.0
        from robosuite.devices.oculus import Oculus

        if not args.dual_arm:
            device = Oculus(
                env=env,
                pos_sensitivity=args.pos_sensitivity,
                rot_sensitivity=args.rot_sensitivity,
            )
        else:
            device = []
            for arm_index, arm in enumerate(["left", "right"]):
                device.append(
                    Oculus(
                        env=env,
                        pos_sensitivity=args.pos_sensitivity,
                        rot_sensitivity=args.rot_sensitivity,
                        arm_index=arm_index,
                    )
                )
    else:
        raise ValueError
    if not isinstance(device, list):
        device = [device]
    print(len(device), "devices")

    # make a new timestamped directory
    new_dir = os.path.join(args.directory)
    assert not os.path.exists(
        os.path.join(new_dir, "demo.hdf5")
    ), "Directory already exists"
    os.makedirs(new_dir, exist_ok=True)

    excluded_eps = []

    # collect demonstrations
    valid_demos = 0
    while True:
        print()
        ep_directory, discard_traj = collect_human_trajectory(
            env,
            device,
            args.arm,
            args.config,
            mirror_actions,
            render=(args.renderer != "mjviewer"),
            max_fr=args.max_fr,
            env_name=config["env_name"],
        )
        # ask the human if they want to keep the trajectory
        print("Valid demos so far:", valid_demos)
        if confirm_user("Do you want to keep this trajectory? (y/n)"):
            discard_traj = False
        else:
            discard_traj = True

        if not discard_traj:
            valid_demos += 1
        if not args.debug:
            if discard_traj and ep_directory is not None:
                excluded_eps.append(ep_directory.split("/")[-1])
            if discard_traj:
                continue
            hdf5_path = gather_demonstrations_as_hdf5(
                tmp_directory, new_dir, env_info, excluded_episodes=excluded_eps
            )
            convert_to_robomimic_format(hdf5_path)
        if not confirm_user("continue collecting demos? (y/n)"):
            break
        if valid_demos >= args.n_demos:
            break
