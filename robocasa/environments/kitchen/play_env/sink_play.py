import os
from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU

PLAY_TASK_NAME_TO_CLASS = {
    "turn_on_faucet": "TurnOnFaucetDebug",
    "turn_off_faucet": "TurnOffFaucetDebug",
    "close_cabinet": "CloseCabDebug",
    "open_cabinet": "OpenCabDebug",
    "pnp_sink_to_plate": "PnPSinkToPlateDebug",
    "pnp_plate_to_sink": "PnPPlateToSinkDebug",
    "pnp_counter_to_plate": "PnPCounterToPlateDebug",
    "pnp_counter_to_cabinet": "PnPCounterToCabDebug",
    "pnp_counter_to_sink": "PnPCounterToSinkDebug",
    "l1_pnp_sink_to_plate": "SinkPlayEnv_PnPSinkToPlayL1",
    "l2_pnp_sink_to_plate": "SinkPlayEnv_PnPSinkToPlayL2",
    "l3_pnp_sink_to_plate": "SinkPlayEnv_PnPSinkToPlayL3",
    "l1_pnp_plate_to_sink": "SinkPlayEnv_PnPPlateToSinkL1",
    "l2_pnp_plate_to_sink": "SinkPlayEnv_PnPPlateToSinkL2",
    "l3_pnp_plate_to_sink": "SinkPlayEnv_PnPPlateToSinkL3",
}

# copied from ArrangeVegetables
class SinkBase(Kitchen):
    """
    Arrange Vegetables: composite task for Chopping Food activity.

    Simulates the task of arranging vegetables on the cutting board.

    Steps:
        Take the vegetables from the sink and place them on the cutting board.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.45, 0.55))
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "There is no language here."
        return ep_meta

    def _get_obj_cfgs(self):
        raise NotImplementedError

    def _check_success(self):
        return False


class SinkPlayEnvDebug(SinkBase):
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab_id = self.register_fixture_ref("cab", dict(id=FixtureType.CABINET_TOP))

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="right", top_size=(0.2, 0.2)
                    ),
                    size=(0.30, 0.25),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="packed_food",
                obj_groups="packaged_food",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="right", top_size=(0.1, 0.65)
                    ),
                    size=(0.3, 0.35),
                    pos=("ref", -1.0),
                    rotation=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
                ),
            )
        )
        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(1.0, 1.0),
                    rotation=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
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
        return cfgs

    def _reset_observables(self):
        super()._reset_observables()
        for jnt_name in self.sim.model.joint_names:
            if ("cab" in jnt_name) and ("right" in jnt_name):
                joint_idx = self.sim.model.joint_names.index(jnt_name)
                joint_value = 1.8
                self.sim.data.qpos[self.sim.model.jnt_qposadr[joint_idx]] = joint_value
        # for jnt_name in self.sim.model.joint_names:
        #     if "container" in jnt_name:
        #         # get the object and its id
        #         obj_name = '_'.join(jnt_name.split("_")[:-1])
        #         elem = self.objects[obj_name].get_obj()
        #         xml = self.sim.model.get_xml()
        #         with open("model.xml", "w") as f:
        #             f.write(xml)


class SinkPlayEnv(SinkPlayEnvDebug):
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab_id = self.register_fixture_ref(
            "cab_main", dict(id=FixtureType.CABINET_TOP)
        )

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        cfgs.append(
            dict(
                name="fruit",
                obj_groups="fruit_set_1",
                placement=dict(
                    fixture=self.cab_id,
                    size=(0.25, 0.25),
                    pos=(0.75, 0.05),
                    offset=(0.0, 0.0),
                ),
            )
        )
        for i, cfg in enumerate(cfgs):
            if cfg["name"] == "vegetable":
                cfgs[i]["obj_groups"] = "vegetable_set_1"
            if cfg["name"] == "plate":
                cfgs[i] = dict(
                    name="plate",
                    obj_groups="plate",
                    graspable=False,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.sink, loc="right", top_size=(0.2, 0.2)
                        ),
                        size=(0.30, 0.25),
                        pos=("ref", -1.0),
                    ),
                )

        return cfgs


class SinkPlayEnvTrain(SinkPlayEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SinkPlayEnvVal(SinkPlayEnv):
    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        for i, cfg in enumerate(cfgs):
            if cfg["name"] == "vegetable":
                cfgs[i]["obj_groups"] = "vegetable_set_2"
            if cfg["name"] == "plate":
                cfgs[i] = dict(
                    name="plate",
                    obj_groups="bowl",
                    graspable=False,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.sink, loc="right", top_size=(0.2, 0.2)
                        ),
                        size=(0.30, 0.25),
                        pos=("ref", -1.0),
                    ),
                )
        return cfgs


class SinkPlayEnv_PnPSinkToPlayL1(SinkPlayEnvTrain):
    @property
    def imp_obj_name(self):
        return "vegetable"

    def start_container_name(self):
        return "vegetable_container"

    def end_container_name(self):
        return "plate"

    @property
    def name(self):
        return "l1_pnp_sink_to_plate"

    def _reset_internal(self):
        super()._reset_internal()
        # assert OU.check_obj_fixture_contact(self, self.start_container_name(), self.sink)
        # assert OU.check_obj_in_receptacle(self, self.imp_obj_name, self.start_container_name())
        # assert OU.check_obj_fixture_contact(self, self.end_container_name(), self.counter)

    def _check_success(self):
        return (
            OU.check_obj_in_receptacle(
                self, self.imp_obj_name, self.end_container_name()
            )
            and OU.gripper_obj_far(self, obj_name=self.imp_obj_name)
            and OU.check_obj_fixture_contact(
                self, self.end_container_name(), self.counter
            )
        )

    def meta_file_names():
        config_names = ["ep_meta_001.json", "ep_meta_003.json", "ep_meta_004.json"]
        return config_names

    def meta_file_path():
        return [
            os.path.join(
                os.environ["CASAPLAY_DATAROOT"],
                "../scene_configs/SinkPlayEnvTrain/training",
                config_name,
            )
            for config_name in SinkPlayEnv_PnPSinkToPlayL1.meta_file_names()
        ]


class SinkPlayEnv_PnPSinkToPlayL2(SinkPlayEnv_PnPSinkToPlayL1):
    @property
    def name(self):
        return "l2_pnp_sink_to_plate"

    def meta_file_names():
        # config_names = ["ep_meta_001.json", "ep_meta_003.json", "ep_meta_004.json"]
        config_names = ["ep_meta_016.json", "ep_meta_024.json", "ep_meta_004.json"]
        return config_names

    def meta_file_path():
        return [
            os.path.join(
                os.environ["CASAPLAY_DATAROOT"],
                "../scene_configs/SinkPlayEnvVal/l2",
                config_name,
            )
            for config_name in SinkPlayEnv_PnPSinkToPlayL2.meta_file_names()
        ]


class SinkPlayEnv_PnPSinkToPlayL3(SinkPlayEnv_PnPSinkToPlayL1):
    @property
    def name(self):
        return "l3_pnp_sink_to_plate"

    def meta_file_names():
        config_names = ["ep_meta_008.json", "ep_meta_009.json", "ep_meta_019.json"]
        return config_names

    def meta_file_path():
        return [
            os.path.join(
                os.environ["CASAPLAY_DATAROOT"],
                "../scene_configs/SinkPlayEnvVal/l3",
                config_name,
            )
            for config_name in SinkPlayEnv_PnPSinkToPlayL3.meta_file_names()
        ]


class EvaluateSinkPlayEnvDebug(SinkPlayEnvDebug):
    @property
    def name(self):
        raise NotImplementedError

    def meta_file_path():
        return [
            os.path.join(
                os.environ["CASAPLAY_DATAROOT"],
                "../scene_configs/SinkPlayEnvDebug/training",
                config_name,
            )
            for config_name in ["ep_meta_000.json"]
        ]

    def check_cabinet_open(self, cab_name):
        cab_state = self.fixtures[cab_name].get_door_state(self)
        return cab_state > 0.05


class TurnOnFaucetDebug(EvaluateSinkPlayEnvDebug):
    @property
    def name(self):
        return "turn_on_faucet"

    def _reset_internal(self):
        super()._reset_internal()
        self.sink.set_handle_state(self, self.rng, mode="off")

    def _check_success(self):
        handle_state = self.sink.get_handle_state(self)
        water_on = handle_state["water_on"]
        return water_on


class TurnOffFaucetDebug(EvaluateSinkPlayEnvDebug):
    @property
    def name(self):
        return "turn_off_faucet"

    def _reset_internal(self):
        super()._reset_internal()
        self.sink.set_handle_state(self, self.rng, mode="on")

    def _check_success(self):
        handle_state = self.sink.get_handle_state(self)
        water_on = handle_state["water_on"]
        return not water_on


# class CloseCabDebug(EvaluateSinkPlayEnvDebug):
#     @property
#     def name(self):
#         return "close_cabinet"
#     def _check_success(self):
#         # some about main group
#         all_cabinets = [name for name in self.fixtures.keys() if 'cab_' in name]
#         all_cabinets = [name for name in all_cabinets if 'main' in name]
#         # import ipdb; ipdb.set_trace()
#         assert len(all_cabinets) == 1
#         return not self.check_cabinet_open(all_cabinets[0])
# class OpenCabDebug(EvaluateSinkPlayEnvDebug):
#     @property
#     def name(self):
#         return "open_cabinet"
#     def _check_success(self):
#         # some about main group
#         all_cabinets = [name for name in self.fixtures.keys() if 'cab_' in name]
#         all_cabinets = [name for name in all_cabinets if 'main' in name]
#         assert len(all_cabinets) == 1
#         return self.check_cabinet_open(all_cabinets[0])
class PnPSinkToPlateDebug(EvaluateSinkPlayEnvDebug):
    @property
    def name(self):
        return "pnp_sink_to_plate"

    def _check_success(self):
        obj_name = "vegetable"
        tar_name = "plate"
        is_in = OU.check_obj_in_receptacle(self, obj_name, tar_name)
        is_tar_contact = OU.check_obj_fixture_contact(self, tar_name, self.counter)
        return is_in and OU.gripper_obj_far(self, obj_name=obj_name) and is_tar_contact


class PnPPlateToSinkDebug(EvaluateSinkPlayEnvDebug):
    @property
    def name(self):
        return "pnp_plate_to_sink"

    def _check_success(self):
        obj_name = "vegetable"
        # tar_name = 'vegetable_container'
        # is_in = OU.check_obj_in_receptacle(self, obj_name, tar_name)
        is_in = OU.check_obj_fixture_contact(self, obj_name, self.sink)
        return is_in and OU.gripper_obj_far(self, obj_name=obj_name)


class PnPCounterToPlateDebug(EvaluateSinkPlayEnvDebug):
    @property
    def name(self):
        return "pnp_counter_to_plate"

    def _check_success(self):
        obj_name = "packed_food"
        tar_name = "plate"
        is_in = OU.check_obj_in_receptacle(self, obj_name, tar_name)
        is_tar_contact = OU.check_obj_fixture_contact(self, tar_name, self.counter)
        return is_in and OU.gripper_obj_far(self, obj_name=obj_name) and is_tar_contact


# class PnPCounterToCabDebug(EvaluateSinkPlayEnvDebug):
#     @property
#     def name(self):
#         return "pnp_counter_to_cabinet"
#     def _check_success(self):
#         obj_name = 'packed_food'
#         all_cabinets = [name for name in self.fixtures.keys() if 'cab_' in name]
#         is_in = False
#         for cab_name in all_cabinets:
#             cab_obj = self.fixtures[cab_name]
#             is_in = OU.obj_inside_of(self, obj_name, cab_obj)
#             if is_in:
#                 break
#         return is_in and OU.gripper_obj_far(self, obj_name=obj_name)
class PnPCounterToSinkDebug(EvaluateSinkPlayEnvDebug):
    @property
    def name(self):
        return "pnp_counter_to_sink"

    def _check_success(self):
        obj_name = "packed_food"
        tar_name = "vegetable_container"
        is_in = OU.check_obj_in_receptacle(self, obj_name, tar_name)
        is_tar_contact = OU.check_obj_fixture_contact(self, tar_name, self.sink)
        # is_in = OU.check_obj_fixture_contact(self, obj_name, self.sink)
        return is_in and OU.gripper_obj_far(self, obj_name=obj_name) and is_tar_contact


class ArrangeVegetablesPlayEnv(SinkBase):
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab_id = self.register_fixture_ref("cab", dict(id=FixtureType.CABINET_TOP))

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="right", top_size=(0.2, 0.2)
                    ),
                    size=(0.30, 0.25),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="packed_food",
                obj_groups="packaged_food",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="right", top_size=(0.1, 0.65)
                    ),
                    size=(0.1, 0.45),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="receptacle",
                obj_groups="receptacle",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.30, 0.20),
                    pos=(-1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.30, 0.20),
                    pos=(1.0, 1.0),
                ),
            )
        )
        return cfgs

    def _reset_observables(self):
        super()._reset_observables()
        for jnt_name in self.sim.model.joint_names:
            # print(jnt_name)
            if ("cab" in jnt_name) and ("right" in jnt_name):
                joint_idx = self.sim.model.joint_names.index(jnt_name)
                joint_value = 1.8
                self.sim.data.qpos[self.sim.model.jnt_qposadr[joint_idx]] = joint_value
        # if "cab_2_left_group_doorhinge" in self.sim.model.joint_names:
        #     print("cab_2_left_group_doorhinge exists")
        #     joint_idx = self.sim.model.joint_names.index("cab_2_left_group_doorhinge")
        #     joint_value = 1.8
        #     self.sim.data.qpos[self.sim.model.jnt_qposadr[joint_idx]] = joint_value


class VeggieOnPlateDebug(ArrangeVegetablesPlayEnv):
    def _check_success(self):
        veggie_plate_contact = OU.check_obj_in_receptacle(self, "vegetable", "plate")
        return veggie_plate_contact

    def meta_file_path(self):
        return "/home/rutavms/research/gaze/icrt/robocasa/scene_configs/ArrangeVegetablesPlayEnv/training/ep_meta_000.json"


class KettleOnPlateTrain(ArrangeVegetablesPlayEnv):
    def _check_success(self):
        kettle_plate_contact = OU.check_obj_in_receptacle(self, "receptacle", "plate")
        return kettle_plate_contact

    def meta_file_path(self):
        return "/home/rutavms/research/gaze/icrt/robocasa/scene_configs/ArrangeVegetablesPlayEnv/training/ep_meta_000.json"


class VeggieOnPlateTrain(ArrangeVegetablesPlayEnv):
    def _check_success(self):
        veggie_plate_contact = OU.check_obj_in_receptacle(self, "vegetable", "plate")
        return veggie_plate_contact

    def meta_file_path(self):
        return "/home/rutavms/research/gaze/icrt/robocasa/scene_configs/ArrangeVegetablesPlayEnv/training/ep_meta_001.json"


class PackedFoodOnPlateTrain(ArrangeVegetablesPlayEnv):
    def _check_success(self):
        veggie_plate_contact = OU.check_obj_in_receptacle(self, "packed_food", "plate")
        return veggie_plate_contact

    def meta_file_path(self):
        return "/home/rutavms/research/gaze/icrt/robocasa/scene_configs/ArrangeVegetablesPlayEnv/training/ep_meta_002.json"


class KettleOnPlate(ArrangeVegetablesPlayEnv):
    def _check_success(self):
        kettle_plate_contact = OU.check_obj_in_receptacle(self, "receptacle", "plate")
        return kettle_plate_contact

    def meta_file_path(self):
        return "/home/rutavms/research/gaze/icrt/robocasa/scene_configs/ArrangeVegetablesPlayEnv/validation/ep_meta_000.json"


class VeggieOnPlate(ArrangeVegetablesPlayEnv):
    def _check_success(self):
        veggie_plate_contact = OU.check_obj_in_receptacle(self, "vegetable", "plate")
        return veggie_plate_contact

    def meta_file_path(self):
        return "/home/rutavms/research/gaze/icrt/robocasa/scene_configs/ArrangeVegetablesPlayEnv/validation/ep_meta_001.json"


class PackedFoodOnPlate(ArrangeVegetablesPlayEnv):
    def _check_success(self):
        veggie_plate_contact = OU.check_obj_in_receptacle(self, "packed_food", "plate")
        return veggie_plate_contact

    def meta_file_path(self):
        return "/home/rutavms/research/gaze/icrt/robocasa/scene_configs/ArrangeVegetablesPlayEnv/validation/ep_meta_003.json"
