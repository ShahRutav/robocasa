from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import create_env, run_random_rollouts
from robocasa.scripts.collect_demos import collect_human_trajectory

import numpy as np

# choose random task
env = create_env(
    env_name="HeatMug",
    render_onscreen=True,
    layout_ids=[1],
    style_ids=[0],
    # renderer="mjviewer",
    seed=0, # set seed=None to run unseeded
)

# from robosuite.devices import SpaceMouse

# device = SpaceMouse(
#     env=env,
#     pos_sensitivity=4.0,
#     rot_sensitivity=4.0,
#     # vendor_id=macros.SPACEMOUSE_VENDOR_ID,
#     # product_id=macros.SPACEMOUSE_PRODUCT_ID,
# )

from robosuite.devices import Keyboard

device = Keyboard(env=env, pos_sensitivity=4.0, rot_sensitivity=4.0)


ep_directory, discard_traj = collect_human_trajectory(
    env,
    device,
    "right",
    "single-arm-opposed",
    mirror_actions=True,
    render=False,
    max_fr=30,
    print_info=False,
)

