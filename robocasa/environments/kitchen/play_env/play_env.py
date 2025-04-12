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
        if self.layout_id not in [3]:
            try:
                self.cab_sink = self.register_fixture_ref(
                    "cab_sink", dict(id=FixtureType.CABINET, ref=self.sink, loc="above")
                )
            except:
                import ipdb

                ipdb.set_trace()
        else:
            self.cab_sink = -1
        # self.EXCLUDE_LAYOUTS = [8,9] # only used for testing in L3

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "There is no language here."
        return ep_meta

    def _remove_specific_fixtures(self):
        remove_keys = ["toaster_main_group"]
        keys = list(self.fixtures.keys())
        for key in keys:
            if key in remove_keys:
                del self.fixtures[key]
        for elem in self.fixture_cfgs:
            if elem["name"] in remove_keys:
                self.fixture_cfgs.remove(elem)
        return

    def _get_obj_cfgs(self):
        raise NotImplementedError

    def _check_success(self):
        return False


class SinkEnvForPlay(BaseEnvForPlay):
    EXCLUDE_LAYOUTS = [3]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.init_robot_base_pos = self.sink
        return

    def _remove_specific_fixtures(self):
        remove_keys = [
            "coffee_machine_left_group",
            "coffee_machine_right_group",
            "coffee_machine_main_group",
        ]
        keys = list(self.fixtures.keys())
        for key in keys:
            if key in remove_keys:
                del self.fixtures[key]
        for elem in self.fixture_cfgs:
            if elem["name"] in remove_keys:
                self.fixture_cfgs.remove(elem)
        return

    def _get_obj_cfgs(self):
        cfgs = []
        """
        cfgs.append(
            dict(
                name="fruit",
                obj_groups="fruit_set_train",
                obj_registries=("objaverse", "aigen"),
                obj_instance_split=None,
                graspable=True,
                placement=dict(
                    fixture=self.counter_sink,
                    size=(1.0, 1.0),
                    rotation=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
                    pos=("ref", -1.0),
                    try_to_place_in="container_set_train",
                    container_kwargs=dict(
                        placement=dict(
                            fixture=self.counter_sink,
                            sample_region_kwargs=dict(
                                ref=self.sink, loc="left_right", top_size=(0.25, 0.25)
                            ),
                            size=(0.35, 0.45),
                            pos=("ref", -1.0),
                        ),
                    ),
                ),
            )
        )
        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable_set_train",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(1.0, 1.0),
                    rotation=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
                    pos=("ref", -1.0),
                    try_to_place_in="container_set_train",
                    container_kwargs=dict(
                        placement=dict(
                            fixture=self.sink,
                            size=(0.25, 0.25),
                            pos=(0.5, 0.4),
                        ),
                    ),
                ),
            )
        )
        if self.cab_sink == -1:
            import ipdb; ipdb.set_trace()
            # pass
            # cfgs.append(
            #     dict(
            #         name="meat",
            #         obj_groups=["meat_set_train", "bread_set_train"],
            #         graspable=True,
            #         max_size=(0.15, 0.15, None),
            #         obj_registries=("objaverse", "aigen"),
            #         obj_instance_split=None,
            #         placement=dict(
            #             fixture=self.counter,
            #             sample_region_kwargs=dict(
            #                 ref=self.sink, loc="left_right", top_size=(0.2, 0.2)
            #             ),
            #             pos=("ref", -1.0),
            #             size=(0.35, 0.45),
            #         ),
            #     )
            # )
        else:
            cfgs.append(
                dict(
                    name="meat",
                    obj_groups=["meat_set_train", "bread_set_train"],
                    graspable=True,
                    max_size=(0.15, 0.15, None),
                    obj_registries=("objaverse", "aigen"),
                    obj_instance_split=None,
                    placement=dict(
                        fixture=self.cab_sink,
                        size=(0.45, 0.15),
                        pos=(0.75, 0.05),
                        offset=(-0.15, -0.05),
                    ),
                )
            )
        """
        return cfgs

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        if self.cab_sink != -1:
            self.cab_sink.set_door_state(min=0.9, max=1.0, env=self, rng=self.rng)


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
        # randomly either placed the packaged item or the vegetable
        rnd = self.rng.uniform()

        if (rnd < 0.5) or (self.layout_id in [1, 3, 6]):
            cfgs.append(
                dict(
                    name="packed_food",
                    obj_groups="packaged_food_train",
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
                            (-3 * np.pi / 8, -np.pi / 4),
                            (np.pi / 4, 3 * np.pi / 8),
                        ],
                    ),
                )
            )
        if (rnd > 0.5) or (self.layout_id in [1, 3, 6]):
            cfgs.append(
                dict(
                    name="vegetable",
                    obj_groups="vegetable_set_train",
                    graspable=True,
                    placement=dict(
                        fixture=self.counter_stove,
                        size=(1.0, 1.0),
                        rotation=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
                        pos=("ref", -1.0),
                        try_to_place_in="container_set_train",
                        container_kwargs=dict(
                            placement=dict(
                                fixture=self.counter_stove,
                                sample_region_kwargs=dict(
                                    ref=self.stove,
                                    loc="left_right",
                                    top_size=(0.35, 0.35),
                                ),
                                size=(0.35, 0.55),
                                pos=("ref", -1.0),
                            ),
                        ),
                    ),
                )
            )
        cfgs.append(
            dict(
                name="obj",
                obj_groups="fruit_set_train",
                graspable=True,
                obj_registries=("objaverse", "aigen"),
                obj_instance_split=None,
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )
        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat_set_train",
                graspable=True,
                max_size=(0.15, 0.15, None),
                obj_registries=("objaverse", "aigen"),
                obj_instance_split=None,
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                    try_to_place_in="pan",
                ),
            )
        )
        return cfgs
