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

macros.SHOW_SITES = True


class MultiTaskBase(Kitchen):
    EXCLUDE_LAYOUTS = []

    def __init__(self, *args, **kwargs):
        # check what is returned in args and kwargs
        super().__init__(*args, **kwargs)

    def _reset_internal(self):
        self._n_steps = 0
        return super()._reset_internal()

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter_stove = self.register_fixture_ref(
            "counter_stove",
            dict(id=FixtureType.COUNTER, ref=self.stove),
        )
        self.counter_stove_right = self.register_fixture_ref(
            "counter_stove_right",
            dict(id=FixtureType.COUNTER, ref=self.stove, loc="right"),
        )
        self.counter_stove_left = self.register_fixture_ref(
            "counter_stove_left",
            dict(id=FixtureType.COUNTER, ref=self.stove, loc="left"),
        )
        self.counter_sink = self.register_fixture_ref(
            "counter_sink",
            dict(id=FixtureType.COUNTER, ref=self.sink),
        )
        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )
        self.counter_microwave = self.register_fixture_ref(
            "counter_microwave",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        # cabinet next to the microwave
        self.cabinet_counter_microwave = self.register_fixture_ref(
            "cabinet_counter_microwave",
            dict(id=FixtureType.CABINET, ref=self.counter_microwave, loc="above"),
        )
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

    def _check_stove_on(self, knob):
        knob_on = False
        knobs_state = self.stove.get_knobs_state(env=self)
        knob_value = knobs_state[knob]
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

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang(obj_name="fruit")
        ep_meta["lang"] = f"Pick up the {obj_lang} and place it in the sink."
        return ep_meta

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
        return

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


class MemFruitPickLeftFar(MemFruitInSinkLeftFar):
    def _reset_internal(self):
        super()._reset_internal()
        self.orig_z_dist = OU.get_z_dist(
            self, obj_name="fruit", container_name="fruit_container"
        )
        return

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang(obj_name="fruit")
        ep_meta["lang"] = f"Pick up the {obj_lang}."
        return ep_meta

    def _check_success(self):
        # check if the fruit is away from the plate by 4 cm more than the original z distance
        return (
            OU.get_z_dist(self, obj_name="fruit", container_name="fruit_container")
            > self.orig_z_dist + 0.08
        )


class MemFruitPickRightFar(MemFruitInSinkRightFar):
    def _reset_internal(self):
        super()._reset_internal()
        self.orig_z_dist = OU.get_z_dist(
            self, obj_name="fruit", container_name="fruit_container"
        )
        return

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang(obj_name="fruit")
        ep_meta["lang"] = f"Pick up the {obj_lang}."
        return ep_meta

    def _check_success(self):
        return (
            OU.get_z_dist(self, obj_name="fruit", container_name="fruit_container")
            > self.orig_z_dist + 0.08
        )


class MemHeatPot(MultiTaskBase):
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.init_robot_base_pos = self.stove
        return

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang(obj_name="meat")
        wait_time_in_mins = self.stove_wait_timer_threshold / kobject.COOK_FPS / 60
        ep_meta[
            "lang"
        ] = f"Turn on the stove, cook the {obj_lang}, wait for {wait_time_in_mins:.1f} minutes, and turn off the stove."
        return ep_meta

    @property
    def pan_location_on_stove(self):
        return "front_left"

    def _reset_internal(self):
        super()._reset_internal()
        self.turn_on_stove_success = False
        self.stove_wait_timer = 0
        self.turn_off_stove_success = False
        self.knob = self.pan_location_on_stove

        # get the object name from the sampled object
        meat_cat = self.find_object_cfg_by_name("meat")["info"]["cat"]
        cook_time = kobject.OBJ_COOK_TIMINGS[meat_cat]
        self.stove_wait_timer_threshold = cook_time
        # just one minute more than the cook time
        self.stove_wait_timer_max_threshold = (
            self.stove_wait_timer_threshold + 60 * kobject.COOK_FPS
        )
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
        location_dist_list.sort(key=lambda x: x[1])
        location, dist = location_dist_list[0]
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
                            rotation=[(-np.pi / 2 - np.pi / 8, -np.pi / 2 + np.pi / 8)],
                            sample_region_kwargs=dict(
                                locs=[self.pan_location_on_stove],
                            ),
                        ),
                        # rotation=[
                        #     (np.pi / 2 - np.pi / 16, np.pi / 2 + np.pi / 16),
                        # ],
                    ),
                ),
            )
        )
        return cfgs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        is_stove_on = self._check_stove_on(self.knob)
        if not self.turn_on_stove_success:  # we have not turned on the stove yet
            if is_stove_on:
                print("*" * 100)
                print("stove turned on")
                self.turn_on_stove_success = True
                self.stove_wait_timer = 0
        elif is_stove_on:  # we have turned on the stove and it is still on
            self.count_empty_actions = True
            self.stove_wait_timer += 1
            print(
                f"stove wait timer: {self.stove_wait_timer}/{self.stove_wait_timer_threshold}"
            )

        # we know that the stove is on, and we have waited for a while but not too much, so we can turn it off
        if (
            self.turn_on_stove_success
            and (self.stove_wait_timer > self.stove_wait_timer_threshold)
            and (self.stove_wait_timer < self.stove_wait_timer_max_threshold)
        ):  # we have turned on the stove and it has been on for a while
            self.count_empty_actions = False  # stop recording empty actions
            if macros.SHOW_SITES:  # only during debugging or data collection
                print("CLOSE STOVE!!!!!")
            self.turn_off_stove_success = not self._check_stove_on(self.knob)
            if macros.SHOW_SITES:
                print("stove turned off")
                print("*" * 100)
        return obs, reward, done, info

    def _check_success(self):
        return (
            self.turn_on_stove_success
            and self.stove_wait_timer > self.stove_wait_timer_threshold
            and self.turn_off_stove_success
        )


class MemHeatPotMultiple(MultiTaskBase):
    """
    Goal is to track multiple objects on the stove during the episode
    """

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.init_robot_base_pos = self.stove
        return

    @property
    def pan_location_on_stove(self):
        return "front_left"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang(obj_name="meat")
        veggie_lang = self.get_obj_lang(obj_name="vegetable")
        wait_time_in_mins = self.stove_wait_timer_threshold / kobject.COOK_FPS / 60
        veggie_add_time_in_mins = self.veggie_add_time_threshold / kobject.COOK_FPS / 60
        ep_meta[
            "lang"
        ] = f"Turn on the stove with the {obj_lang}, add the {veggie_lang} after {veggie_add_time_in_mins:.1f} minutes, and wait for {wait_time_in_mins:.1f} minutes before turning off the stove."
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()
        self.stove_wait_timer = 0
        self.veggie_add_success = False
        self.veggie_add_time = 0  # initialize to a negative value to indicate that the veggie has not been added yet
        self.turn_on_stove_success = False
        self.turn_off_stove_success = False
        self.knob = self.pan_location_on_stove

        # get the object name from the sampled object
        meat_cat = self.find_object_cfg_by_name("meat")["info"]["cat"]
        veggie_cat = self.find_object_cfg_by_name("vegetable")["info"]["cat"]
        cook_time = kobject.OBJ_COOK_TIMINGS[meat_cat]
        veggie_add_time = kobject.OBJ_WAIT_TIMINGS[veggie_cat]
        self.stove_wait_timer_threshold = cook_time
        self.veggie_add_time_threshold = veggie_add_time
        # just one minute more than the cook time
        self.stove_wait_timer_max_threshold = (
            self.stove_wait_timer_threshold + 60 * kobject.COOK_FPS
        )
        # just one minute more than the veggie add time
        self.veggie_add_time_max_threshold = (
            self.veggie_add_time_threshold + 60 * kobject.COOK_FPS
        )
        return

    def _get_obj_cfgs(self):
        split_type = self.split_type()
        cfgs = []
        cfgs.append(
            dict(
                name="meat",
                obj_groups=f"meat_with_minimum_three_minutes",
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
                            rotation=[(-np.pi / 2 - np.pi / 8, -np.pi / 2 + np.pi / 8)],
                            sample_region_kwargs=dict(
                                locs=[self.pan_location_on_stove],
                            ),
                        ),
                    ),
                ),
            )
        )
        # vegetables_after_x_minutes with each one after one min, two mins, three mins in stove_counter_left
        cfgs.append(
            dict(
                name="vegetable",
                obj_groups=f"vegetables_after_x_minutes",
                obj_registries=("objaverse", "aigen"),
                graspable=True,
                obj_scale=0.8,
                placement=dict(
                    fixture=self.counter_stove_left,
                    size=(0.15, 0.15),
                    pos=("ref", -1.0),
                    sample_region_kwargs=dict(
                        ref=self.counter_stove_left,
                    ),
                    offset=(0.4, 0.05),
                ),
            )
        )

        return cfgs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        is_stove_on = self._check_stove_on(self.knob)
        if not self.turn_on_stove_success:  # we have not turned on the stove yet
            if is_stove_on:  # check if just turned it on
                print("*" * 100)
                print("stove turned on")
                self.turn_on_stove_success = True
                self.stove_wait_timer = 0  # start the timer for the meat
        elif is_stove_on:  # we have turned on the stove and it is still on
            self.count_empty_actions = True
            self.stove_wait_timer += 1
            if macros.SHOW_SITES:
                if not self.veggie_add_success:
                    print(
                        f"veggie timer: {self.stove_wait_timer}/{self.veggie_add_time_threshold}; stove timer: {self.stove_wait_timer}/{self.stove_wait_timer_max_threshold}"
                    )
                else:
                    print(
                        f"stove timer: {self.stove_wait_timer}/{self.stove_wait_timer_threshold}"
                    )

        # if stove is on and the veggie add timer is within the threshold
        self._update_veggie_add_time()
        if (
            self.turn_on_stove_success
            and not self.veggie_add_success
            and (
                self.stove_wait_timer >= self.veggie_add_time_threshold
            )  # stove is on after the veggie add time threshold
            and (
                self.stove_wait_timer <= self.veggie_add_time_max_threshold
            )  # stove is on within the max threshold to add the veggie
        ):
            if (
                macros.SHOW_SITES
            ):  # print to indicate that is is about time to add the veggie
                print("ADD VEGGIE!!!!!")

        # check if the veggie is added. if added, mark it as success.
        if (
            self.turn_on_stove_success
            and not self.veggie_add_success
            and (
                self.veggie_add_time >= self.veggie_add_time_threshold
            )  # give success only if the veggie was added within the two boundaries
            and (self.veggie_add_time <= self.veggie_add_time_max_threshold)
        ):
            self.veggie_add_success = True
            if macros.SHOW_SITES and self.veggie_add_success:
                print("VEGGIE ADDED!!!!!")

        # we know that the stove is on and veggie is added, and we have waited for a while but not too much, so we can turn it off
        if (
            self.turn_on_stove_success
            and self.veggie_add_success
            and (self.stove_wait_timer >= self.stove_wait_timer_threshold)
            and (self.stove_wait_timer <= self.stove_wait_timer_max_threshold)
        ):  # we have turned on the stove and it has been on for a while
            self.count_empty_actions = False  # stop recording empty actions
            if macros.SHOW_SITES:  # only during debugging or data collection
                print("CLOSE STOVE!!!!!")
            self.turn_off_stove_success = not self._check_stove_on(self.knob)
            if macros.SHOW_SITES and self.turn_off_stove_success:
                print("stove turned off")
                print("*" * 100)

        self._n_steps += 1
        return obs, reward, done, info

    def _update_veggie_add_time(self):
        if (
            (not self.veggie_add_success)
            and (self._check_veggie_on_pan())
            and (self.stove_wait_timer < self.veggie_add_time_max_threshold)
            and (self.veggie_add_time == 0)
        ):
            self.veggie_add_time = self.stove_wait_timer
            print(
                f"VEGGIE ADDED AT {self.veggie_add_time} steps after stove was turned on"
            )
        return

    def _check_veggie_on_pan(self):
        return OU.check_obj_in_receptacle(
            self, obj_name="vegetable", receptacle_name="meat_container"
        )

    def _check_success(self):
        return (
            self.turn_on_stove_success
            and self.veggie_add_success
            and self.stove_wait_timer >= self.stove_wait_timer_threshold
            and self.stove_wait_timer <= self.stove_wait_timer_max_threshold
            and self.turn_off_stove_success
        )


class MemWashAndReturn(MultiTaskBase):
    def _reset_internal(self):
        super()._reset_internal()
        self.init_robot_base_pos = self.sink
        self.place_success = False
        self.final_success = False
        joint_val = self.rng.uniform(0.45, 0.50)
        # self.sink.set_handle_state(mode="on", env=self, rng=self.rng)
        self.sim.data.set_joint_qpos(
            "{}handle_joint".format(self.sink.naming_prefix), joint_val
        )
        return

    @property
    def fruit_container_counter_loc(self):
        # to be implemented by the child class
        raise NotImplementedError

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
                                loc=self.fruit_container_counter_loc[0],
                                top_size=(0.25, 0.25),
                            ),
                            size=(0.3, 0.3),
                            pos=("ref", -1.0),
                            offset=(
                                self.fruit_container_counter_loc[1][0],
                                self.fruit_container_counter_loc[1][1],
                            ),  # 0.5 to the right of the counter sink and 0.1 to the top of the sink
                        ),
                    ),
                ),
            )
        )
        cfgs.append(
            dict(
                name="fruit_container2",
                obj_groups=f"container_set_{split_type}",
                obj_registries=("objaverse", "aigen"),
                obj_instance_split=None,
                placement=dict(
                    fixture=self.counter_sink,
                    size=(0.3, 0.3),
                    pos=("ref", -1.0),
                    sample_region_kwargs=dict(
                        loc=self.fruit_container_counter_loc2[0],
                        top_size=(0.25, 0.25),
                        ref=self.sink,
                    ),
                    offset=(
                        self.fruit_container_counter_loc2[1][0],
                        self.fruit_container_counter_loc2[1][1],
                    ),
                ),
            )
        )
        return cfgs

    def _update_success(self):
        # check if the fruit is in the sink
        obj_name = "fruit"
        if not self.place_success:
            self.place_success = OU.check_obj_fixture_contact(self, obj_name, self.sink)
        elif not self.final_success:
            # check if the fruit is back in the container 'fruit_container'
            tar_name = self.destination_container_name
            is_in = OU.check_obj_in_receptacle(self, obj_name, tar_name)
            is_gripper_far = OU.gripper_obj_far(self, obj_name=obj_name)
            self.final_success = is_in and is_gripper_far
        return

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self._update_success()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._update_success()
        return obs, reward, done, info

    def _check_success(self):
        return self.place_success and self.final_success

    def compute_robot_base_placement_pose(self, ref_fixture, offset=None):
        additional_offset = (0.25, 0.00)
        if offset is None:
            offset = additional_offset
        else:
            offset = (
                offset[0] + additional_offset[0],
                offset[1] + additional_offset[1],
            )
        return super().compute_robot_base_placement_pose(ref_fixture, offset)


class MemWashAndReturnSameLocation(MemWashAndReturn):
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
            obj_pos, obj_quat, obj = self.object_placements["fruit"]
            fruit_pos = np.array(obj_pos)

            # Create success region around the original fruit position
            success_region_size = [0.05, 0.05, 0.001]  # 5cm x 5cm x 1mm height

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
                self.sim.data.body_xpos[self.obj_body_id["fruit"]]
            )

        obj_name = "fruit"
        if not self.place_success:
            self.place_success = OU.check_obj_fixture_contact(self, obj_name, self.sink)
        elif not self.final_success:
            # check if the fruit is back in the container 'fruit_container'
            # is_in = OU.check_obj_in_receptacle(self, obj_name, tar_name)
            # check the distance between the fruit and the original position
            # is_in = np.linalg.norm(OU.get_obj_pos(self, obj_name="fruit") - self.orig_fruit_pos) < 0.05 # 5cm
            is_in_x = (
                np.abs(
                    OU.get_obj_pos(self, obj_name="fruit")[0] - self.orig_fruit_pos[0]
                )
                < 0.05
            )
            is_in_y = (
                np.abs(
                    OU.get_obj_pos(self, obj_name="fruit")[1] - self.orig_fruit_pos[1]
                )
                < 0.05
            )
            is_in = is_in_x and is_in_y
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
        cfgs.append(
            dict(
                name="fruit",
                obj_groups=f"fruit_set_{split_type}",
                obj_registries=("objaverse", "aigen"),
                obj_instance_split=None,
                graspable=True,
                placement=dict(
                    fixture=self.counter_sink,
                    size=(0.25, 0.25),
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
        ep_meta[
            "lang"
        ] = f"Wash the fruit and return it to the same location as before."
        return ep_meta


class MemWashAndReturnLeft(MemWashAndReturn):
    @property
    def fruit_container_counter_loc(self):
        return "left", (0.5, 0.05)

    @property
    def fruit_container_counter_loc2(self):
        return "right", (0.0, 0.05)

    @property
    def destination_container_name(self):
        return "fruit_container"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Wash the fruit and return it to the container."
        return ep_meta


class MemWashAndReturnRight(MemWashAndReturn):
    @property
    def fruit_container_counter_loc(self):
        return "right", (0.0, 0.05)

    @property
    def fruit_container_counter_loc2(self):
        return "left", (0.5, 0.05)

    @property
    def destination_container_name(self):
        return "fruit_container"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Wash the fruit and return it to the container."
        return ep_meta


class MemPutKBreadInMicrowave(MultiTaskBase):
    """
    This task is to put the bread in the microwave and close the door.
    """

    fixed_n_bread_samples = None

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.init_robot_base_pos = self.microwave
        self.microwave._turned_on = False
        self.n_objects_inside = 0
        self.max_breads = 3
        self.n_bread_samples = (
            np.random.randint(1, self.max_breads + 1)
            if self.fixed_n_bread_samples is None
            else self.fixed_n_bread_samples
        )
        self.door_id = FixtureType.DOOR_TOP_HINGE_SINGLE
        self.door_fxtr = self.register_fixture_ref("door_fxtr", dict(id=self.door_id))
        self.init_robot_base_pos = self.door_fxtr
        return

    def set_ep_meta(self, ep_meta):
        super().set_ep_meta(ep_meta)
        if "n_bread_samples" in ep_meta:
            self.fixed_n_bread_samples = ep_meta["n_bread_samples"]
        else:
            self.fixed_n_bread_samples = None
        return

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Put all the breads in the microwave and close the microwave door."
        ep_meta["n_bread_samples"] = self.n_bread_samples
        return ep_meta

    def _reset_internal(self):
        # open the door
        self.door_fxtr.set_door_state(min=0.9, max=1.0, env=self, rng=self.rng)
        return super()._reset_internal()

    def _check_success(self):
        # see how many breads are in the microwave
        is_success = True
        self.n_breads_in_microwave = 0
        for obj_name in self.objects:
            if obj_name.startswith("bread_"):
                in_microwave = OU.check_obj_fixture_contact(
                    self, obj_name, self.microwave
                )
                if in_microwave:
                    self.n_breads_in_microwave += 1
        is_success = self.n_breads_in_microwave == self.n_bread_samples

        door_state = self.door_fxtr.get_door_state(env=self)
        for joint_p in door_state.values():
            if joint_p > 0.2:
                is_success = False
                break
        return is_success

    def _get_obj_cfgs(self):
        split_type = self.split_type()
        cfgs = []
        for i in range(self.max_breads):
            placement_reference = (
                self.counter_microwave
                if i < self.n_bread_samples
                else self.counter_sink
            )
            sample_region_reference = (
                self.microwave if i < self.n_bread_samples else self.sink
            )
            sample_size = (0.25, 0.25)
            cfgs.append(
                dict(
                    name=f"bread_{i}",
                    obj_groups=f"bread_set_{split_type}",
                    obj_registries=("objaverse", "aigen"),
                    obj_instance_split=None,
                    graspable=True,
                    placement=dict(
                        fixture=placement_reference,
                        size=sample_size,
                        pos=("ref", -1.0),
                        sample_region_kwargs=dict(
                            loc="left_right",
                            top_size=sample_size,
                            ref=sample_region_reference,
                        ),
                        offset=(0.1, 0.0),
                    ),
                )
            )
        return cfgs

    def compute_robot_base_placement_pose(self, ref_fixture, offset=None):
        additional_offset = (0.65, 0.0)
        if offset is None:
            offset = additional_offset
        else:
            offset = (
                offset[0] + additional_offset[0],
                offset[1] + additional_offset[1],
            )
        return super().compute_robot_base_placement_pose(ref_fixture, offset)


class MemRetrieveOilsFromCounter(MultiTaskBase):
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.init_robot_base_pos = self.stove
        return

    def _reset_internal(self):
        super()._reset_internal()
        self.init_robot_base_pos = self.stove
        self.orig_olive_oil_pos = -1.0
        return

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang(obj_name="olive_oil")
        ep_meta["lang"] = f"Pick up the {obj_lang}."
        return ep_meta

    def _check_success(self):
        if self.orig_olive_oil_pos == -1.0:
            return False
        # print(
        #     "Olive oil pos",
        #     OU.get_obj_pos(self, obj_name="olive_oil")[2],
        #     " orig_olive_oil_pos",
        #     self.orig_olive_oil_pos,
        # )
        return (
            OU.get_obj_pos(self, obj_name="olive_oil")[2]
            > self.orig_olive_oil_pos + 0.08
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if (self._n_steps > 2) and (self.orig_olive_oil_pos == -1.0):
            self.orig_olive_oil_pos = OU.get_obj_pos(self, obj_name="olive_oil")[2]
        self._n_steps += 1
        return obs, reward, done, info

    def _get_obj_cfgs(self):
        split_type = self.split_type()
        cfgs = []
        cfgs.append(
            dict(
                name="canola_oil",
                obj_groups=f"canola_oil",
                obj_registries=("objaverse", "aigen"),
                placement=dict(
                    fixture=self.oil_container_counter_loc[2],
                    size=(0.4, 0.15),
                    pos=("ref", -1.0),
                    object_scale=0.8,  # make this graspable
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc=self.oil_container_counter_loc[0],
                        top_size=(0.25, 0.25),
                    ),
                    offset=self.oil_container_counter_loc[1],
                ),
            )
        )
        cfgs.append(
            dict(
                name="olive_oil",
                obj_groups=f"olive_oil",
                obj_registries=("objaverse", "aigen"),
                placement=dict(
                    fixture=self.oil_container_counter_loc[2],
                    size=(0.4, 0.15),
                    pos=("ref", -1.0),
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc=self.oil_container_counter_loc2[0],
                        top_size=(0.25, 0.25),
                    ),
                    offset=self.oil_container_counter_loc2[1],
                ),
            )
        )
        return cfgs


class MemRetrieveOilsFromCounterLL(MemRetrieveOilsFromCounter):
    """
    This task is to retrieve the oils from the counter and put them in the stove.
    """

    @property
    def oil_container_counter_loc(self):
        return "left", (-0.35, 0.05), self.counter_stove_left

    @property
    def oil_container_counter_loc2(self):
        return "left", (-0.35, 0.05), self.counter_stove_left


class MemRetrieveOilsFromCounterRR(MemRetrieveOilsFromCounter):
    @property
    def oil_container_counter_loc(self):
        return "right", (0.35, 0.05), self.counter_stove_right

    @property
    def oil_container_counter_loc2(self):
        return "right", (0.35, 0.05), self.counter_stove_right


class MemRetrieveOilsFromCounterLR(MemRetrieveOilsFromCounter):
    @property
    def oil_container_counter_loc(self):
        return "left", (-0.35, 0.05), self.counter_stove_left

    @property
    def oil_container_counter_loc2(self):
        return "right", (0.35, 0.05), self.counter_stove_right


class MemRetrieveOilsFromCounterRL(MemRetrieveOilsFromCounter):
    @property
    def oil_container_counter_loc(self):
        return "right", (0.35, 0.05), self.counter_stove_right

    @property
    def oil_container_counter_loc2(self):
        return "left", (-0.35, 0.05), self.counter_stove_left


class MemPutKBowlInCabinet(MultiTaskBase):
    """
    This task is to put the bread in the microwave and close the door.
    """

    fixed_n_bowls_samples = None
    fixed_n_plate_samples = None

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.init_robot_base_pos = (
            self.microwave
        )  # starting position is same as microwave
        self.microwave._turned_on = False
        self.n_objects_inside = 0
        self.max_bowls = 3
        self.max_plate_samples = 2
        self.n_bowls_samples = (
            np.random.randint(1, self.max_bowls + 1)
            if self.fixed_n_bowls_samples is None
            else self.fixed_n_bowls_samples
        )
        self.n_plate_samples = (
            np.random.randint(1, self.max_plate_samples + 1)
            if self.fixed_n_plate_samples is None
            else self.fixed_n_plate_samples
        )
        # starting position is front of the microwave next to the cabinet
        self.init_robot_base_pos = self.cabinet_counter_microwave
        return

    def set_ep_meta(self, ep_meta):
        super().set_ep_meta(ep_meta)
        if "n_bowls_samples" in ep_meta:
            self.fixed_n_bowls_samples = ep_meta["n_bowls_samples"]
        else:
            self.fixed_n_bowls_samples = None
        if "n_plate_samples" in ep_meta:
            self.fixed_n_plate_samples = ep_meta["n_plate_samples"]
        else:
            self.fixed_n_plate_samples = None
        return

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Put all the bowls in the cabinet and close the right cabinet door."
        ep_meta["n_bowls_samples"] = self.n_bowls_samples
        ep_meta["n_plate_samples"] = self.n_plate_samples
        return ep_meta

    def _reset_internal(self):
        # open the door
        self.cabinet_counter_microwave.set_door_state(
            min=0.9, max=1.0, env=self, rng=self.rng
        )
        return super()._reset_internal()

    def _check_success(self):
        # is_success = False
        # # see how many breads are in the microwave
        is_success = True
        self.n_bowls_in_cabinet = 0
        for obj_name in self.objects:
            if obj_name.startswith("bowl_"):
                in_cabinet = OU.check_obj_fixture_contact(
                    self, obj_name, self.cabinet_counter_microwave
                )
                if in_cabinet:
                    self.n_bowls_in_cabinet += 1
        is_success = self.n_bowls_in_cabinet == self.n_bowls_samples
        door_state = self.cabinet_counter_microwave.get_door_state(env=self)
        if door_state["right_door"] > 0.2:  # door is open
            is_success = False
        return is_success

    def _get_obj_cfgs(self):
        split_type = self.split_type()
        cfgs = []
        for i in range(self.max_bowls):
            sample_size = (0.55, 0.35)
            placement_reference = (
                self.counter_microwave
                if i < self.n_bowls_samples
                else self.counter_sink  # some far away placement
            )
            sample_region_reference = (
                self.microwave if i < self.n_bowls_samples else self.sink
            )
            cfgs.append(
                dict(
                    name=f"bowl_{i}",
                    obj_groups=f"bowl",
                    obj_registries=("objaverse", "aigen"),
                    graspable=True,
                    object_scale=0.5,
                    placement=dict(
                        fixture=placement_reference,
                        size=sample_size,
                        pos=("ref", -1.0),
                        sample_region_kwargs=dict(
                            loc="left_right",
                            top_size=sample_size,
                            ref=sample_region_reference,
                        ),
                        offset=(0.1, 0.0),
                    ),
                )
            )
        for i in range(self.max_plate_samples):
            sample_size = (0.75, 0.45)
            placement_reference = (
                self.counter_microwave
                if i < self.n_plate_samples
                else self.counter_sink  # some far away placement
            )
            sample_region_reference = (
                self.microwave if i < self.n_plate_samples else self.sink
            )
            cfgs.append(
                dict(
                    name=f"plate_{i}",
                    obj_groups=f"plate",
                    obj_registries=("objaverse", "aigen"),
                    graspable=False,
                    placement=dict(
                        fixture=placement_reference,
                        size=sample_size,
                        pos=("ref", -1.0),
                        sample_region_kwargs=dict(
                            loc="left_right",
                            top_size=sample_size,
                            ref=sample_region_reference,
                        ),
                        offset=(0.1, 0.0),
                    ),
                )
            )
        return cfgs
