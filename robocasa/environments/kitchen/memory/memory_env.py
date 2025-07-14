import os
import numpy as np

from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU
import robocasa.macros as macros
from robocasa.models.fixtures import *
from termcolor import colored

macros.SHOW_SITES = True


class MultiTaskBase(Kitchen):
    EXCLUDE_LAYOUTS = []

    def __init__(self, *args, **kwargs):
        # check what is returned in args and kwargs
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter_stove = self.register_fixture_ref(
            "counter_stove",
            dict(id=FixtureType.COUNTER, ref=self.stove),
        )
        self.counter_sink = self.register_fixture_ref(
            "counter_sink",
            dict(id=FixtureType.COUNTER, ref=self.sink),
        )
        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )
        print("layout_id: ", self.layout_id)
        if self.layout_id not in [1, 3, 4, 5, 6]:
            self.cab_sink = self.register_fixture_ref(
                "cab_sink", dict(id=FixtureType.CABINET, ref=self.sink, loc="above")
            )
        else:
            self.cab_sink = None

        self.all_drawers = [
            fixture
            for fixture in self.fixtures.values()
            if fixture_is_type(fixture, FixtureType.DRAWER)
        ]

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
            # "toaster_main_group",
            # "toaster_right_group",
            # "toaster_left_group",
            # "coffee_machine_left_group",
            # "coffee_machine_right_group",
            # "coffee_machine_main_group",
        ]
        self.fixtures = {
            key: value for key, value in self.fixtures.items() if key not in remove_keys
        }
        self.fixture_cfgs = [
            elem for elem in self.fixture_cfgs if elem["name"] not in remove_keys
        ]
        return

    def _check_stove_on(self):
        knob_on = False
        if self.knob is not None:
            knobs_state = self.stove.get_knobs_state(env=self)
            knob_value = knobs_state[self.knob]
            knob_on = 0.35 <= np.abs(knob_value) <= 2 * np.pi - 0.35
        return knob_on

    def _get_obj_cfgs(self):
        raise NotImplementedError

    def _check_success(self):
        return False


class MemFruitInSink(MultiTaskBase):
    @property
    def fruit_container_counter_loc(self):
        return "left_right"

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.init_robot_base_pos = self.sink
        return

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
                            size=(0.3, 0.3),
                            pos=("ref", -1.0),
                            offset=(0.5, 0.0),
                        ),
                    ),
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
            self.cab_sink.set_door_state(min=0.0, max=0.01, env=self, rng=self.rng)
        self.sink.set_handle_state(mode="on", env=self, rng=self.rng)
        for drawer in self.all_drawers:
            drawer.set_door_state(min=0.0, max=0.001, env=self, rng=self.rng)

    def _check_success(self):
        obj_name = "fruit"
        is_in = OU.check_obj_fixture_contact(self, obj_name, self.sink)
        return is_in and OU.gripper_obj_far(self, obj_name=obj_name)


class MemFruitInSinkLeftFar(MemFruitInSink):
    @property
    def fruit_container_counter_loc(self):
        return "left"

    def _get_obj_cfgs(self):
        obj_cfgs = super()._get_obj_cfgs()
        for cfg in obj_cfgs:
            if cfg["name"] == "fruit":
                cfg["placement"]["container_kwargs"]["placement"]["offset"] = (
                    -0.5,
                    0.0,
                )
        return obj_cfgs


class MemFruitInSinkRightFar(MemFruitInSink):
    @property
    def fruit_container_counter_loc(self):
        return "right"

    def _get_obj_cfgs(self):
        obj_cfgs = super()._get_obj_cfgs()
        for cfg in obj_cfgs:
            if cfg["name"] == "fruit":
                cfg["placement"]["container_kwargs"]["placement"]["offset"] = (0.5, 0.0)
        return obj_cfgs


class MemHeatPot(MultiTaskBase):
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        return

    @property
    def pan_location_on_stove(self):
        return "front_left"

    def _reset_internal(self):
        super()._reset_internal()
        self.turn_on_stove_success = False
        self.stove_wait_timer = 0
        self.stove_wait_timer_threshold = 300
        self.stove_wait_timer_max_threshold = self.stove_wait_timer_threshold + 200
        self.turn_off_stove_success = False
        # self.knob = self._get_obj_location_on_stove("meat_container", threshold=0.08)
        self.knob = self.pan_location_on_stove
        return

    def _get_obj_location_on_stove(self, obj_name, threshold=0.08):
        """
        Check if the object is on the stove and close to a burner and the knob is on.
        Returns the location of the burner if the object is on the stove, close to a burner, and the burner is on.
        None otherwise.
        """
        # TODO: make sure only one stove is detected or the one with least distance is selected and less than threshold
        knobs_state = self.stove.get_knobs_state(env=self)
        obj = self.objects[obj_name]
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])[0:2]
        print("*" * 20)
        location_dist_list = []
        for location, site in self.stove.burner_sites.items():
            if site is not None:
                burner_pos = np.array(self.sim.data.get_site_xpos(site.get("name")))[
                    0:2
                ]
                dist = np.linalg.norm(burner_pos - obj_pos)
                location_dist_list.append((location, dist))
                print(f"location: {location}, site: {site}, dist: {dist:.3f}")
        location_dist_list.sort(key=lambda x: x[1])
        location, dist = location_dist_list[0]
        print("location: ", location, "dist: ", dist)
        obj_on_site = dist < threshold
        if obj_on_site:
            return location
        else:
            print(
                colored(
                    f"Closest knob is {location} but dist is {dist:.3f} which is greater than threshold {threshold:.2f}",
                    "red",
                )
            )
        return None

    def _get_obj_cfgs(self):
        split_type = self.split_type()
        cfgs = []
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
                    size=(0.1, 0.1),
                    try_to_place_in="pan",
                    container_kwargs=dict(  # this will be overriding the placement fixture & rest will be copied
                        placement=dict(
                            loc=self.pan_location_on_stove,
                            rotation=[(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)],
                            sample_region_kwargs=dict(
                                locs=[self.pan_location_on_stove],
                            ),
                        ),
                    ),
                ),
            )
        )
        return cfgs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        is_stove_on = self._check_stove_on()
        if not self.turn_on_stove_success:  # we have not turned on the stove yet
            if is_stove_on:
                print("*" * 100)
                print("stove turned on")
                self.turn_on_stove_success = True
                self.stove_wait_timer = 0
        elif is_stove_on:  # we have turned on the stove and it is still on
            self.count_empty_actions = True
            self.stove_wait_timer += 1
            print("stove wait timer: ", self.stove_wait_timer)

        # we know that the stove is on, and we have waited for a while but not too much, so we can turn it off
        if (
            self.turn_on_stove_success
            and (self.stove_wait_timer > self.stove_wait_timer_threshold)
            and (self.stove_wait_timer < self.stove_wait_timer_max_threshold)
        ):  # we have turned on the stove and it has been on for a while
            self.count_empty_actions = False  # stop recording empty actions
            print("CLOSE STOVE!!!!!")
            self.turn_off_stove_success = not self._check_stove_on()
            print("stove turned off")
            print("*" * 100)
        return obs, reward, done, info

    def _check_success(self):
        return (
            self.turn_on_stove_success
            and self.stove_wait_timer > self.stove_wait_timer_threshold
            and self.turn_off_stove_success
        )
