class MemHeatPotMultiple(MultiTaskBase):
    """
    now we have two pans, on with meat and the other with veggie. goal is turn on and off both in correct time
    """

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.init_robot_base_pos = self.stove
        return

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang(obj_name="meat")
        wait_time_in_mins = self.stove_wait_timer_threshold / kobject.COOK_FPS / 60

        veggie_lang = self.get_obj_lang(obj_name="vegetable")
        wait_time_veggie_in_mins = (
            self.stove_wait_timer_threshold_2 / kobject.COOK_FPS / 60
        )
        ep_meta[
            "lang"
        ] = f"Turn on the stoves, cook the {obj_lang} for {wait_time_in_mins:.1f} minutes, and {veggie_lang} for {wait_time_veggie_in_mins:.1f} minutes."
        return ep_meta

    @property
    def pan_location_on_stove(self):
        return "front_left"

    @property
    def pan_location_on_stove_2(self):
        return "front_right"

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

        self.turn_on_stove_success_2 = False
        self.stove_wait_timer_2 = 0
        self.turn_off_stove_success_2 = False
        self.knob_2 = self.pan_location_on_stove_2

        # get the veggie add time
        veggie_cat = self.find_object_cfg_by_name("vegetable")["info"]["cat"]
        veggie_cook_time = kobject.OBJ_WAIT_TIMINGS[veggie_cat]
        self.stove_wait_timer_threshold_2 = veggie_cook_time
        self.stove_wait_timer_max_threshold_2 = (
            self.stove_wait_timer_threshold_2 + 60 * kobject.COOK_FPS
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
        cfgs.append(
            dict(
                name="vegetable",
                obj_groups=f"vegetable_with_maximum_three_minutes",
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
                            loc=self.pan_location_on_stove_2,
                            rotation=[(-np.pi / 2 - np.pi / 8, -np.pi / 2 + np.pi / 8)],
                            sample_region_kwargs=dict(
                                locs=[self.pan_location_on_stove_2],
                            ),
                        ),
                        rotation=[
                            (np.pi / 2 - np.pi / 16, np.pi / 2 + np.pi / 16),
                        ],
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
            if macros.SHOW_SITES:
                print(
                    f"stove wait timer: {self.stove_wait_timer}/{self.stove_wait_timer_threshold}"
                )

        is_stove_on_2 = self._check_stove_on(self.knob_2)
        if not self.turn_on_stove_success_2:  # we have not turned on the stove yet
            if is_stove_on_2:
                print("*" * 100)
                print("stove 2 turned on")
                self.turn_on_stove_success_2 = True
                self.stove_wait_timer_2 = 0
        elif is_stove_on_2:  # we have turned on the stove and it is still on
            self.count_empty_actions = True
            self.stove_wait_timer_2 += 1
            if macros.SHOW_SITES:
                print(
                    f"stove 2 wait timer: {self.stove_wait_timer_2}/{self.stove_wait_timer_threshold_2}"
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
        if (
            self.turn_on_stove_success_2  # this will always close before
            and (self.stove_wait_timer_2 > self.stove_wait_timer_threshold_2)
            and (self.stove_wait_timer_2 < self.stove_wait_timer_max_threshold_2)
        ):  # we have turned on the stove and it has been on for a while
            if macros.SHOW_SITES:  # only during debugging or data collection
                print("CLOSE STOVE 2!!!!!")
            self.turn_off_stove_success_2 = not self._check_stove_on(self.knob_2)
            if macros.SHOW_SITES:
                print("stove 2 turned off")
                print("*" * 100)
        return obs, reward, done, info

    def _check_success(self):
        return (
            self.turn_on_stove_success
            and self.turn_on_stove_success_2
            and self.turn_off_stove_success
            and self.turn_off_stove_success_2
        )
