import os
import numpy as np

from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU
import robocasa.macros as macros

macros.SHOW_SITES = False


class BaseEnvForPlay(Kitchen):
    EXCLUDE_LAYOUTS = []

    def __init__(self, *args, **kwargs):
        # check what is returned in args and kwargs
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter_stove = self.register_fixture_ref(
            # "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
            "counter_stove",
            dict(id=FixtureType.COUNTER, ref=self.stove),
        )
        self.counter_sink = self.register_fixture_ref(
            # "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=[0.30, 0.40])
            "counter_sink",
            dict(id=FixtureType.COUNTER, ref=self.sink),
        )
        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )
        print("layout_id: ", self.layout_id)
        # print(f"self.sink {self.sink}; self.stove {self.stove}; self.counter_stove {self.counter_stove}; self.counter_sink {self.counter_sink}; self.microwave {self.microwave}")
        if self.layout_id not in [1, 3, 4, 5, 6]:
            self.cab_sink = self.register_fixture_ref(
                "cab_sink", dict(id=FixtureType.CABINET, ref=self.sink, loc="above")
            )
        else:
            self.cab_sink = None

    def split_type(self):
        """
        Returns the split type of the environment.
        """
        return "train"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "There is no language here."
        return ep_meta

    def _remove_specific_fixtures(self):
        remove_keys = [
            "toaster_main_group",
            "toaster_right_group",
            "toaster_left_group",
            "coffee_machine_left_group",
            "coffee_machine_right_group",
            "coffee_machine_main_group",
        ]
        self.fixtures = {
            key: value for key, value in self.fixtures.items() if key not in remove_keys
        }
        self.fixture_cfgs = [
            elem for elem in self.fixture_cfgs if elem["name"] not in remove_keys
        ]
        return

    def _get_obj_cfgs(self):
        raise NotImplementedError

    def _check_success(self):
        return False


class SinkEnvForPlay(BaseEnvForPlay):
    EXCLUDE_LAYOUTS = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.init_robot_base_pos = self.sink
        return

    @property
    def fruit_container_counter_loc(self):
        return "left_right"

    @property
    def bread_container_counter_loc(self):
        return "left_right"

    @property
    def packed_food_counter_loc(self):
        return "left_right"

    def _get_obj_cfgs(self):
        split_type = self.split_type()
        cfgs = []
        cfgs.append(
            dict(
                name="fruit",
                obj_groups=f"fruit_set_{split_type}",
                obj_registries=("objaverse", "aigen"),
                obj_instance_split=None,
                graspable=True,
                placement=dict(
                    fixture=self.counter_sink,
                    size=(1.0, 1.0),
                    rotation=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
                    pos=("ref", -1.0),
                    try_to_place_in=f"container_set_{split_type}",
                    container_kwargs=dict(
                        placement=dict(
                            fixture=self.counter_sink,
                            sample_region_kwargs=dict(
                                ref=self.sink,
                                loc=self.fruit_container_counter_loc,
                                top_size=(0.25, 0.25),
                            ),
                            size=(0.35, 0.35),
                            pos=("ref", -1.0),
                        ),
                    ),
                ),
            )
        )
        if self.layout_id not in [7]:
            cfgs.append(
                dict(
                    name="bread",
                    obj_groups=f"bread_set_{split_type}",
                    obj_registries=("objaverse", "aigen"),
                    obj_instance_split=None,
                    graspable=True,
                    placement=dict(
                        fixture=self.counter_sink,
                        size=(1.0, 1.0),
                        rotation=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
                        pos=("ref", -1.0),
                        try_to_place_in=f"container_set_{split_type}",
                        container_kwargs=dict(
                            placement=dict(
                                fixture=self.counter_sink,
                                sample_region_kwargs=dict(
                                    ref=self.sink,
                                    loc=self.bread_container_counter_loc,
                                    top_size=(0.25, 0.25),
                                ),
                                size=(0.35, 0.35),
                                pos=("ref", -1.0),
                            ),
                        ),
                    ),
                )
            )
        cfgs.append(
            dict(
                name="packed_food",
                obj_groups=f"packaged_food_{split_type}",
                graspable=True,
                obj_registries=("objaverse", "aigen"),
                obj_instance_split=None,
                placement=dict(
                    fixture=self.counter_sink,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc=self.packed_food_counter_loc,
                        top_size=(0.25, 0.35),
                    ),
                    size=(0.25, 0.35),
                    pos=("ref", -1.0),
                    rotation=[
                        (np.pi / 2 - np.pi / 16, np.pi / 2 + np.pi / 16),
                    ],
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups=f"vegetable_set_{split_type}",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(1.0, 1.0),
                    rotation=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
                    pos=("ref", -1.0),
                    try_to_place_in=f"container_set_{split_type}",
                    container_kwargs=dict(
                        placement=dict(
                            fixture=self.sink,
                            size=(0.3, 0.3),
                            pos=(0.5, 0.4),
                        ),
                    ),
                ),
            )
        )
        if self.cab_sink is not None:
            offset = (-0.15, -0.05)
            size = (0.45, 0.15)
            if self.layout_id == 8:  # make it reachable for each layout
                offset = (-0.5, -0.05)
                size = (0.30, 0.15)
            if self.layout_id == 9:  # make it reachable for each layout
                offset = (0.10, -0.05)
                size = (0.30, 0.15)
            cfgs.append(
                dict(
                    name="meat",
                    obj_groups=[f"meat_set_{split_type}", f"bread_set_{split_type}"],
                    graspable=True,
                    max_size=(0.15, 0.15, None),
                    obj_registries=("objaverse", "aigen"),
                    obj_instance_split=None,
                    placement=dict(
                        fixture=self.cab_sink,
                        size=size,
                        pos=(0.75, 0.05),
                        offset=offset,
                    ),
                )
            )
        else:
            cfgs.append(
                dict(
                    name="meat",
                    obj_groups=f"meat_set_{split_type}",
                    graspable=True,
                    max_size=(0.15, 0.15, None),
                    obj_registries=("objaverse", "aigen"),
                    obj_instance_split=None,
                    placement=dict(
                        fixture=self.counter_sink,
                        sample_region_kwargs=dict(
                            ref=self.sink, loc="left_right", top_size=(0.25, 0.3)
                        ),
                        size=(0.25, 0.30),
                        pos=("ref", -1.0),
                        rotation=[
                            (np.pi / 2 - np.pi / 16, np.pi / 2 + np.pi / 16),
                        ],
                    ),
                )
            )
        return cfgs

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        if self.cab_sink is not None:
            self.cab_sink.set_door_state(min=0.9, max=1.0, env=self, rng=self.rng)
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)


class PnPRightCounterPlateToSink(SinkEnvForPlay):
    @property
    def fruit_container_counter_loc(self):
        return "right"

    @property
    def bread_container_counter_loc(self):
        return "left"

    def set_ep_meta(self, ep_meta):
        ## here we will overrite the object_cfgs if present in the ep_meta. specifically overrite the fruit_container_counter_loc and bread_container_counter_loc
        if "object_cfgs" in ep_meta:
            for obj_cfg in ep_meta["object_cfgs"]:
                if obj_cfg["name"] == "fruit_container":
                    obj_cfg["placement"]["sample_region_kwargs"][
                        "loc"
                    ] = self.fruit_container_counter_loc
                if obj_cfg["name"] == "bread_container":
                    obj_cfg["placement"]["sample_region_kwargs"][
                        "loc"
                    ] = self.bread_container_counter_loc
        super().set_ep_meta(ep_meta)
        return ep_meta

    def _check_success(self):
        obj_name = "fruit"
        is_in = OU.check_obj_fixture_contact(self, obj_name, self.sink)
        return is_in and OU.gripper_obj_far(self, obj_name=obj_name)


class PnPSinkToRightCounterPlate(SinkEnvForPlay):
    @property
    def fruit_container_counter_loc(self):
        return "right"

    @property
    def bread_container_counter_loc(self):
        return "left"

    def _check_success(self):
        obj_name = "vegetable"
        tar_name = "fruit_container"
        is_in = OU.check_obj_in_receptacle(self, obj_name, tar_name)
        is_tar_contact = OU.check_obj_fixture_contact(self, tar_name, self.counter_sink)
        return is_in and OU.gripper_obj_far(self, obj_name=obj_name) and is_tar_contact

    def set_ep_meta(self, ep_meta):
        ## here we will overrite the object_cfgs if present in the ep_meta. specifically overrite the fruit_container_counter_loc and bread_container_counter_loc
        if "object_cfgs" in ep_meta:
            for obj_cfg in ep_meta["object_cfgs"]:
                if obj_cfg["name"] == "fruit_container":
                    obj_cfg["placement"]["sample_region_kwargs"][
                        "loc"
                    ] = self.fruit_container_counter_loc
                if obj_cfg["name"] == "bread_container":
                    obj_cfg["placement"]["sample_region_kwargs"][
                        "loc"
                    ] = self.bread_container_counter_loc
        super().set_ep_meta(ep_meta)
        return ep_meta


class CloseRightCabinetDoor(SinkEnvForPlay):
    def _check_success(self):
        cab_state = self.cab_sink.get_door_state(self)
        return cab_state["right_door"] < 0.1


class CloseLeftCabinetDoor(SinkEnvForPlay):
    def _check_success(self):
        cab_state = self.cab_sink.get_door_state(self)
        return cab_state["left_door"] < 0.1


class StoveEnvForPlay(BaseEnvForPlay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.init_robot_base_pos = self.stove
        return

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.microwave.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        split_type = self.split_type()
        cfgs.append(
            dict(
                name="packed_food",
                obj_groups=f"packaged_food_{split_type}",
                graspable=True,
                obj_registries=("objaverse", "aigen"),
                obj_instance_split=None,
                placement=dict(
                    fixture=self.counter_stove,
                    sample_region_kwargs=dict(
                        ref=self.stove, loc="left_right", top_size=(0.25, 0.35)
                    ),
                    size=(0.25, 0.35),
                    pos=("ref", -1.0),
                    rotation=[
                        (np.pi / 2 - np.pi / 16, np.pi / 2 + np.pi / 16),
                    ],
                ),
            )
        )
        cfgs.append(
            dict(
                name="bread",
                obj_groups=f"bread_set_{split_type}",
                graspable=True,
                obj_registries=("objaverse", "aigen"),
                obj_instance_split=None,
                placement=dict(
                    fixture=self.counter_stove,
                    sample_region_kwargs=dict(
                        ref=self.stove, loc="left_right", top_size=(0.25, 0.35)
                    ),
                    size=(0.25, 0.35),
                    pos=("ref", -1.0),
                    rotation=[
                        (np.pi / 2 - np.pi / 16, np.pi / 2 + np.pi / 16),
                    ],
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups=f"vegetable_set_{split_type}",
                graspable=True,
                placement=dict(
                    fixture=self.counter_stove,
                    size=(1.0, 1.0),
                    rotation=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
                    pos=("ref", -1.0),
                    try_to_place_in=f"container_set_{split_type}",
                    container_kwargs=dict(
                        placement=dict(
                            fixture=self.counter_stove,
                            sample_region_kwargs=dict(
                                ref=self.stove,
                                loc="left_right",
                                top_size=(0.35, 0.35),
                            ),
                            size=(0.35, 0.35),
                            pos=("ref", -1.0),
                        ),
                    ),
                ),
            )
        )
        cfgs.append(
            dict(
                name="fruit",
                obj_groups=f"fruit_set_{split_type}",
                graspable=True,
                obj_registries=("objaverse", "aigen"),
                obj_instance_split=None,
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    offset=(0.0, -0.10),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )
        cfgs.append(
            dict(
                name="meat",
                obj_groups=f"meat_set_{split_type}",
                graspable=True,
                max_size=(0.15, 0.15, None),
                obj_registries=("objaverse", "aigen"),
                obj_instance_split=None,
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                    try_to_place_in="pan",
                    container_kwargs=dict(  # this will be overriding the placement fixture & rest will be copied
                        placement=dict(
                            rotation=[(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)],
                        ),
                    ),
                ),
            )
        )
        return cfgs


class StoveEnvForPlayTest(StoveEnvForPlay):
    def split_type(self):
        return "test"

    def _remove_specific_fixtures(self):
        remove_keys = []
        self.fixtures = {
            key: value for key, value in self.fixtures.items() if key not in remove_keys
        }
        self.fixture_cfgs = [
            elem for elem in self.fixture_cfgs if elem["name"] not in remove_keys
        ]
        return


class SinkEnvForPlayTestV2(SinkEnvForPlay):
    def split_type(self):
        return "test"

    def _remove_specific_fixtures(self):
        # does not remove any fixtures
        return

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        remove_keys = ["bread", "bread_container"]
        cfgs = [cfg for cfg in cfgs if cfg["name"] not in remove_keys]
        return cfgs

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        if self.cab_sink is not None:
            self.cab_sink.set_door_state(min=0.0, max=0.1, env=self, rng=self.rng)


class SinkEnvForPlayTest(SinkEnvForPlay):
    # removes all the fixtures like toaster, coffee machine, etc.
    def split_type(self):
        return "test"
