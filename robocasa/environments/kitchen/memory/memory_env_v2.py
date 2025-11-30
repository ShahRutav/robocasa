import os
from re import S
import numpy as np

from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU
import robocasa.macros as macros
from robocasa.models.fixtures import *
from termcolor import colored

# from robocasa.models.objects.kitchen_objects import OBJ_COOK_TIMINGS
import robocasa.models.objects.kitchen_objects as kobject
from .memory_env import *

# macros.SHOW_SITES = True


def check_if_object_in_success_region(env, obj_name, start_pos, success_region_size):
    is_in_x = (
        np.abs(OU.get_obj_pos(env, obj_name=obj_name)[0] - start_pos[0])
        < success_region_size[0]
    )
    is_in_y = (
        np.abs(OU.get_obj_pos(env, obj_name=obj_name)[1] - start_pos[1])
        < success_region_size[1]
    )
    is_in = is_in_x and is_in_y
    return is_in


class MemWashAndReturnSameLocationV2(MemWashAndReturn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fruit_name = "fruit_0"
        self.success_region_size = [0.05, 0.05, 0.001]

    @property
    def fruit_container_counter_loc(self):
        return "right", (0.05, 0.05)

    def _reset_internal(self):
        super()._reset_internal()
        self.orig_fruit_pos = None
        return

    def _setup_kitchen_references(self):
        # add the fruit sampler goal reference if macros.SHOW_SITES = True
        super()._setup_kitchen_references()
        return

    def _load_model(self, ind=0):
        super()._load_model(ind)

        def _get_success_region_fruit():
            # Use the object_placements from the model - much simpler than XML parsing!
            obj_pos, obj_quat, obj = self.object_placements[self.fruit_name]
            fruit_pos = np.array(obj_pos)

            # Create success region around the original fruit position
            success_region_size = self.success_region_size  # 5cm x 5cm x 1mm height

            # For visualization, use the object position directly (no transformation needed)
            pos_to_vis = deepcopy(fruit_pos)
            return fruit_pos, success_region_size, pos_to_vis

        if macros.SHOW_SITES and hasattr(self, "sim"):
            # if the model already has the success_region_fruit site, update its location
            if (
                hasattr(self.model, "worldbody")
                and hasattr(self.model.worldbody, "site")
                and "success_region_fruit" in self.model.worldbody.site
            ):
                fruit_pos, success_region_size, pos_to_vis = _get_success_region_fruit()
                site = self.model.worldbody.site["success_region_fruit"]
                site.pos = array_to_string(pos_to_vis)
                site.size = array_to_string(success_region_size)
            else:
                # create the success_region_fruit site
                fruit_pos, success_region_size, pos_to_vis = _get_success_region_fruit()
                site_str = """<site type="box" rgba="0 1 0 0.4" size="{size}" pos="{pos}" name="success_region_fruit"/>""".format(
                    pos=array_to_string(pos_to_vis),
                    size=array_to_string(success_region_size),
                )
                site_tree = ET.fromstring(site_str)
                self.model.worldbody.append(site_tree)
        return

    def _update_success(self):
        # check if the fruit is in the sink
        if (self._n_steps > 2) and (self.orig_fruit_pos is None):
            # Get the fruit position from simulation data using obj_body_id
            self.orig_fruit_pos = np.array(
                self.sim.data.body_xpos[self.obj_body_id[self.fruit_name]]
            )

        obj_name = self.fruit_name
        if not self.place_success:
            self.place_success = OU.check_obj_fixture_contact(self, obj_name, self.sink)
        elif not self.final_success:
            is_in = check_if_object_in_success_region(
                self, obj_name, self.orig_fruit_pos, [0.05, 0.05, 0.001]
            )
            is_gripper_far = OU.gripper_obj_far(self, obj_name=obj_name)
            self.final_success = is_in and is_gripper_far
        return

    def _check_success(self):
        return self.place_success and self.final_success

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._update_success()
        self._n_steps += 1
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self._update_success()
        self._n_steps = 0
        return obs

    def _get_obj_cfgs(self):
        split_type = self.split_type()
        cfgs = []
        n_fruit_samples = 5
        for i in range(n_fruit_samples):
            cfgs.append(
                dict(
                    name=f"fruit_{i}",
                    obj_groups=f"fruit_set_{split_type}",
                    obj_registries=("objaverse", "aigen"),
                    obj_instance_split=None,
                    graspable=True,
                    keep_cat_unique=True,
                    placement=dict(
                        fixture=self.counter_sink,
                        size=(0.25, 0.6),
                        pos=("ref", -1.0),
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc=self.fruit_container_counter_loc[0],
                            top_size=(0.3, 0.6),
                        ),
                        offset=self.fruit_container_counter_loc[1],
                    ),
                )
            )
        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        # our goal is to wash and return the fruit 0.
        obj_lang = self.get_obj_lang(obj_name="fruit_0")
        ep_meta[
            "lang"
        ] = f"Wash the {obj_lang} and return it to the same location as before."
        return ep_meta


class MemWashAndReturnSameContainerV2(MemWashAndReturn):
    def __init__(self, *args, **kwargs):
        self.fruit_name = "fruit_0"
        self.destination_container_name = f"{self.fruit_name}_container"
        self.n_extra_containers = 2
        self.n_extra_fruits = 2
        self.n_fruits_with_containers = 2
        super().__init__(*args, **kwargs)

    def split_type(self):
        return "train"

    def _get_obj_cfgs(self):
        split_type = self.split_type()
        cfgs = []
        common_fruit_kwargs = dict(
            obj_groups=f"fruit_set_{split_type}",
            obj_registries=("objaverse", "aigen"),
            obj_instance_split=None,
            graspable=True,
        )
        common_container_kwargs = dict(
            obj_groups=f"container_set_{split_type}",
            obj_registries=("objaverse", "aigen"),
            obj_instance_split=None,
            object_scale=0.5,
        )
        size = (0.3, 0.6)
        top_size = (0.3, 0.6)

        placement_kwargs = dict(
            fixture=self.counter_sink,
            offset=(0.05, 0.05),  # leave 5cm gap
            size=size,
            pos=("ref", -1.0),
            sample_region_kwargs=dict(
                ref=self.sink,
                loc="right",
                top_size=top_size,
            ),
        )
        fruit_index = 0
        container_index = 0
        for i in range(self.n_fruits_with_containers):
            cfgs.append(
                dict(
                    name=f"fruit_{fruit_index}",
                    keep_cat_unique=True,
                    **common_fruit_kwargs,
                    placement=dict(
                        fixture=self.counter_sink,
                        size=(1.0, 1.0),
                        pos=("ref", -1.0),
                        try_to_place_in=f"container_set_{split_type}",
                        container_kwargs=dict(
                            **common_container_kwargs,
                            placement=dict(
                                **placement_kwargs,
                            ),
                        ),
                    ),
                )
            )
            fruit_index += 1
            container_index += 1

        for i in range(self.n_extra_fruits):
            cfgs.append(
                dict(
                    name=f"fruit_{fruit_index}",
                    keep_cat_unique=True,
                    **common_fruit_kwargs,
                    placement=dict(
                        **placement_kwargs,
                    ),
                )
            )
            fruit_index += 1
        for i in range(self.n_extra_containers):
            cfgs.append(
                dict(
                    name=f"container_{container_index}",
                    **common_container_kwargs,
                    placement=dict(
                        **placement_kwargs,
                    ),
                )
            )
            container_index += 1
        return cfgs

    def _update_success(self):
        # check if the fruit is in the sink
        obj_name = "fruit_0"
        if not self.place_success:
            self.place_success = OU.check_obj_fixture_contact(self, obj_name, self.sink)
        elif not self.final_success:
            # check if the fruit is back in the container 'fruit_container'
            tar_name = self.destination_container_name
            is_in = OU.check_obj_in_receptacle(self, obj_name, tar_name)
            is_gripper_far = OU.gripper_obj_far(self, obj_name=obj_name)
            self.final_success = is_in and is_gripper_far
        return

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang(obj_name="fruit_0")
        ep_meta["lang"] = f"Wash {obj_lang} and return it to the same container."
        return ep_meta
