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

# Function to replace robot tags in the XML
def replace_robot_tag(new_model_xml, old_model_xml):
    """
    new_model_xml: str (XML content of the new model)
    old_model_xml: str (XML content of the old model)

    1. Delete all tags starting with 'robot0' in the <actuator> and <asset> tags in the new model.
    2. Add all 'robot0' related tags from the old model into the new model in <actuator> and <asset>.
    3. Replace 'robot0_floating_base' body tag with 'robot0_base' from the old model (including its content).

    Returns:
        str: Modified XML content as a string.
    """

    # Parse XML content
    new_root = ET.fromstring(new_model_xml)
    old_root = ET.fromstring(old_model_xml)

    # Function to filter and transfer robot0 elements
    def transfer_robot0_elements(new_parent, old_parent):
        # Remove existing robot0 elements in new_parent
        if new_parent is not None:
            for elem in list(new_parent):
                if "robot0" in elem.attrib.get("name", ""):
                    new_parent.remove(elem)

        if old_parent is not None:
            if new_parent is None:
                # add a new parent element if it doesn't exist to the
                # Add robot0 elements from old_parent to new_parent as a child of <mujoco> tag
                new_parent = ET.SubElement(new_root, old_parent.tag)
            for elem in old_parent:
                if "robot0" in elem.attrib.get("name", ""):
                    new_parent.append(elem)

    # Update <actuator> elements
    new_actuator = new_root.find("actuator")
    old_actuator = old_root.find("actuator")
    transfer_robot0_elements(new_actuator, old_actuator)

    # Update <asset> elements
    new_asset = new_root.find("asset")
    old_asset = old_root.find("asset")
    transfer_robot0_elements(new_asset, old_asset)

    new_contact = new_root.find("contact")
    old_contact = old_root.find("contact")
    transfer_robot0_elements(new_contact, old_contact)

    new_robot0_base = None
    for body in new_root.findall(".//body"):
        if body.attrib.get("name") == "robot0_floating_base":
            new_robot0_base = body
            break

    # Locate "robot0_floating_base" in old model
    old_robot0_floating_base = None
    for body in old_root.findall(".//body"):
        if body.attrib.get("name") == "robot0_base":
            old_robot0_floating_base = body
            break

    assert new_robot0_base is not None, "robot0_floating_base must exist in new model"
    assert old_robot0_floating_base is not None, "robot0_base must exist in old model"
    # Replace the contents of "robot0_base" with "robot0_floating_base" if both exist
    if new_robot0_base is not None and old_robot0_floating_base is not None:
        # Clear the current children of new_robot0_base
        new_robot0_base.clear()

        # Copy all attributes from old_robot0_floating_base to new_robot0_base
        new_robot0_base.attrib = old_robot0_floating_base.attrib

        # Copy all child elements
        for elem in old_robot0_floating_base:
            new_robot0_base.append(elem)

    # copy the geom name from old model to new model named robot0_floor in the worldbody tag
    # we want to put it in the same location in the new model
    new_worldbody = new_root.find("worldbody")
    old_worldbody = old_root.find("worldbody")
    for geom in old_worldbody.findall(".//geom"):
        if geom.attrib.get("name") == "robot0_floor":
            index = list(old_worldbody).index(geom)
            new_worldbody.insert(index, geom)

    # Convert back to string
    return ET.tostring(new_root, encoding="unicode")


def reset_to_diff_robot(env, state, replace_robot_joints=False, change_to_gr1=False):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml

    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = state["ep_meta"]
            if isinstance(ep_meta, str):
                ep_meta = json.loads(ep_meta)
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        else:
            raise ValueError(
                "Environment does not have set_ep_meta or set_attrs_from_ep_meta function"
            )
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        # we need to first update state["model"] to replace the robot tag with the current robot
        curr_xml = env.sim.model.get_xml()
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        if change_to_gr1:
            xml = replace_robot_tag(new_model_xml=xml, old_model_xml=curr_xml)

        env.reset_from_xml_string(xml)
        env.sim.reset()
        if replace_robot_joints:
            env.robots[0].reset()
            env.sim.forward()

    if "states" in state:
        if replace_robot_joints:
            print("Replacing robot joints")
            robot_indices = env.robots[0]._ref_joint_pos_indexes
            other_indices = set(
                range(env.sim.get_state().qpos.flatten().shape[0])
            ) - set(robot_indices)
            other_indices = sorted(list(other_indices))

            non_robot_qpos_idx = state["non_robot_qpos_idx"]
            assert len(non_robot_qpos_idx) == len(
                other_indices
            ), f"Mismatch in non_robot_qpos_idx: {len(non_robot_qpos_idx)} != {len(other_indices)}"
            qpos_state = state["qpos_state"]
            time = env.sim.data.time
            qvel = env.sim.get_state().qvel.flatten().copy()
            qpos = env.sim.get_state().qpos.flatten().copy()
            # copy over everything except the robot state
            for curr_idx, state_idx in zip(other_indices, non_robot_qpos_idx):
                qpos[curr_idx] = qpos_state[state_idx]
            curr_state = MjSimState(qpos=qpos, qvel=qvel, time=time)
            env.sim.set_state_from_flattened(curr_state.flatten())
            env.sim.forward()
            should_ret = True
        else:
            env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True
        if replace_robot_joints:
            env.robots[0].reset()
            env.sim.forward()

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    return None


def reset_to(env, state):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml

    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = state["ep_meta"]
            if isinstance(ep_meta, str):
                ep_meta = json.loads(ep_meta)
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    # if should_ret:
    #     # only return obs if we've done a forward call - otherwise the observations will be garbage
    #     return get_observation()
    return None


def load_controller_config(controller, robot, control_type, ref_frame):
    if controller == "OSC_POSE":
        return load_part_controller_config(default_controller="OSC_POSE")
    if robot == "PandaOmron":
        raise ValueError(
            "Composite controller is not the one used for Robocasa default dataset. Use OSC_POSE for PandaOmron robots --controller OSC_POSE"
        )

    controller_config = load_composite_controller_config(
        controller=controller,
        robot=robot,
    )
    if "right" in controller_config["body_parts"]:
        controller_config["body_parts"]["right"]["input_type"] = control_type
        controller_config["body_parts"]["right"]["input_ref_frame"] = ref_frame
    if "left" in controller_config["body_parts"]:
        controller_config["body_parts"]["left"]["input_type"] = control_type
        controller_config["body_parts"]["left"]["input_ref_frame"] = ref_frame
    if "WHOLE_BODY" in controller_config["type"]:
        controller_config["composite_controller_specific_configs"][
            "ik_input_ref_frame"
        ] = ref_frame
        controller_config["composite_controller_specific_configs"][
            "ik_input_type"
        ] = control_type
    return controller_config


def get_env_args_from_dataset(dataset_path):
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    env_args = (
        json.loads(f["data"].attrs["env_args"])
        if "data" in f
        else json.loads(f.attrs["env_args"])
    )
    if isinstance(env_args, str):
        env_args = json.loads(env_args)  # double leads to dict type
    f.close()
    # convert env_args to EasyDict
    return env_args


def get_env_meta_from_dataset(dataset_path, index=0):
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    data = f["data"] if "data" in f else f
    keys = list(data.keys())
    env_meta = data[keys[index]].attrs["ep_meta"]
    env_meta = json.loads(env_meta)
    assert isinstance(env_meta, dict), f"Expected dict type but got {type(env_meta)}"
    return env_meta


def make_env(file_name, keys_info, args, env_meta=None):
    env_args = get_env_args_from_dataset(dataset_path=file_name)
    env_meta = get_env_meta_from_dataset(dataset_path=file_name, index=0)
    dataset_controller_config = env_args["env_kwargs"]["controller_configs"]
    if hasattr(args, "controller"):
        control_type = (
            dataset_controller_config["body_parts"]["right"]["input_type"]
            if "body_parts" in dataset_controller_config
            else None
        )
        if control_type is None:
            control_type = (
                "delta" if "delta" in dataset_controller_config else "absolute"
            )
        ref_frame = (
            dataset_controller_config["body_parts"]["right"]["input_ref_frame"]
            if "body_parts" in dataset_controller_config
            else "base"
        )
        controller_config = load_controller_config(
            controller=args.controller,
            robot=args.robots if isinstance(args.robots, str) else args.robots[0],
            control_type=control_type,
            ref_frame=ref_frame,
        )
    else:
        print("No controller specified. Using default controller from the dataset")
        controller_config = dataset_controller_config

    env_name = env_args["env_name"]
    print("Env name: ", env_name)
    if hasattr(args, "task_name") and args.task_name is not None:
        env_name = PLAY_TASK_NAME_TO_CLASS[args.task_name]
    env_kwargs = env_args["env_kwargs"]

    env_kwargs["env_name"] = env_name
    env_kwargs[
        "ep_meta"
    ] = env_meta  # this should ideally reduce exploration for finding correct set of objects
    if hasattr(args, "robots"):
        env_kwargs["robots"] = (
            args.robots if isinstance(args.robots, list) else [args.robots]
        )
    env_kwargs["controller_configs"] = controller_config
    env_kwargs.pop("has_renderer", None)
    env_kwargs.pop("use_camera_obs", None)
    env_kwargs.pop("renderer", None)
    env_kwargs.pop("camera_segmentations", None)
    # print(f"Env args: {env_args}")

    print("Rendering the environment: ", args.render)
    control_freq = 20 if not hasattr(args, "control_freq") else args.control_freq
    print(f"Control frequency: {control_freq}")
    env = robosuite.make(
        has_renderer=args.render,
        has_offscreen_renderer=not args.render,
        renderer="mjviewer",
        use_camera_obs=True if len(keys_info["image_keys"]) > 0 else False,
        camera_segmentations=keys_info.get("segmentation_level", None),
        control_freq=control_freq,
        **env_kwargs,
    )
    return env, env_kwargs


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
    np.random.seed(args.seed)

    reachability_errors_per_robot = {}
    for robot_run in args.robot_names:
        reachability_errors_per_robot[robot_run] = []
        print("Running robot: ", robot_run)
        args.robots = robot_run
        # (Optional) if you have a Mink IK or other JSON config, load it:
        args.controller = (
            "WHOLE_BODY_MINK_IK" if "GR1" in args.robots else args.controller
        )
        if "GR1" not in args.robots:
            args.control_freq = 20

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

                # The first simulator state sets the environment's initial condition
                initial_state = {
                    "states": states[0],
                    "ep_meta": ep_meta,
                }
                initial_state["model"] = ep_group.attrs["model_file"]
                initial_state["non_robot_qpos_idx"] = ep_group.attrs.get(
                    "non_robot_qpos_idx", None
                )
                initial_state["qpos_state"] = ep_group.attrs.get("initial_qpos", None)

                # 4) reset_to(...) from casa_utils
                reset_to_diff_robot(
                    env,
                    initial_state,
                    replace_robot_joints=False,
                    change_to_gr1="GR1" in args.robots,
                )
                env.sim.forward()
                print("done initializing robot at the same position")

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
