from dataclasses import dataclass, field

import numpy as np


PANDA_END_EFFECTOR_INDEX = 11
PANDA_NUM_DOFS = 7
LOWER_LIMITS = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
UPPER_LIMITS = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
JOINT_RANGES = [UPPER_LIMITS[index] - LOWER_LIMITS[index] for index in range(PANDA_NUM_DOFS)]
REST_POSES = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    position: tuple[float, float, float] = (0.60, 0.00, 0.90)
    target: tuple[float, float, float] = (0.60, 0.00, 0.00)
    up: tuple[int, int, int] = (0, 1, 0)
    fov: int = 60
    near: float = 0.1
    far: float = 2.0


@dataclass(frozen=True)
class VisionConfig:
    capture_repeats: int = 3
    top_percentile: float = 85.0
    replan_position_threshold: float = 0.01
    refine_accept_max_shift: float = 0.008


@dataclass(frozen=True)
class SceneConfig:
    plane_urdf: str = "plane.urdf"
    robot_urdf: str = "franka_panda/panda.urdf"
    object_urdf: str = "cube_small.urdf"
    robot_base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    object_position: tuple[float, float, float] = (0.5, 0.1, 0.2)
    use_fixed_base: bool = True
    cam_to_world_bias: np.ndarray = field(default_factory=lambda: np.array([-0.015, -0.015, -0.025]))


@dataclass(frozen=True)
class RuntimeConfig:
    realtime_sleep: bool = True
    sleep_dt: float = 1 /480.0


@dataclass(frozen=True)
class GripperConfig:
    joints: tuple[int, int] = (9, 10)
    open_position: float = 0.04
    default_close_position: float = 0.01
    close_force: float = 30.0
    open_force: float = 80.0
    min_grasp_force: float = 12.0
    hold_force_min: float = 12.0
    motor_force_min: float = 20.0
    force_scale: float = 1000.0
    contact_force_threshold: float = 0.5
    release_contact_force: float = 8.0
    boost_step: float = 2.0
    relax_step: float = 1.0


@dataclass(frozen=True)
class MotionConfig:
    settle_steps: int = 60
    pre_grasp_offset: float = 0.18
    grasp_offset: float = 0.02
    lift_offset: float = 0.25
    validation_lift_offset: float = 0.06
    approach_steps: int = 360
    lift_steps: int = 360
    hold_steps: int = 30
    close_steps: int = 80
    close_stabilize_steps: int = 40
    reach_check_interval: int = 40
    reach_check_max_extra_steps: int = 320
    home_steps: int = 240
    default_orientation_euler: tuple[float, float, float] = (np.pi, 0.0, 0.0)
    candidate_yaws: tuple[float, ...] = (
        -3 * np.pi / 4,
        -np.pi / 2,
        -np.pi / 4,
        0.0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4,
        np.pi,
    )
    grasp_xy_search_step: float = 0.01
    grasp_xy_search_radius: float = 0.02
    ik_position_tolerance: float = 0.015
    joint_limit_margin: float = 0.10
    arm_force: float = 500.0
    arm_max_velocity: float = 2.0
    arm_position_gain: float = 0.3
    arm_velocity_gain: float = 1.0
    lift_arm_max_velocity: float = 0.6
    lift_arm_position_gain: float = 0.12
    lift_success_min_height_delta: float = 0.03
    lift_success_max_xy_error: float = 0.08


@dataclass(frozen=True)
class PlaceConfig:
    enabled: bool = False
    release_position: tuple[float, float, float] = (0.55, -0.20, 0.08)
    pre_place_offset: float = 0.14
    retreat_offset: float = 0.20
    settle_steps: int = 90


@dataclass(frozen=True)
class LoggingConfig:
    logger_name: str = "grasp_system"
    level: str = "INFO"
    log_file: str = "logs/grasp_system.log"
    log_vision_frames: bool = False
    log_contact_details: bool = True


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    place: PlaceConfig = field(default_factory=PlaceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
