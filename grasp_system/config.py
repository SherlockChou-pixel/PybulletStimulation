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
class SceneConfig:
    plane_urdf: str = "plane.urdf"
    robot_urdf: str = "franka_panda/panda.urdf"
    object_urdf: str = "cube_small.urdf"
    robot_base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    object_position: tuple[float, float, float] = (0.8, 0.1, 0.2)
    use_fixed_base: bool = True
    cam_to_world_bias: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -0.03]))


@dataclass(frozen=True)
class GripperConfig:
    joints: tuple[int, int] = (9, 10)
    open_position: float = 0.04
    default_close_position: float = 0.01
    open_force: float = 80.0
    close_force: float = 120.0
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
    approach_steps: int = 240
    hold_steps: int = 180
    close_steps: int = 80
    close_stabilize_steps: int = 90
    default_orientation_euler: tuple[float, float, float] = (np.pi, 0.0, 0.0)
    arm_force: float = 500.0
    arm_max_velocity: float = 1.0
    arm_position_gain: float = 0.1
    arm_velocity_gain: float = 1.0


@dataclass(frozen=True)
class LoggingConfig:
    logger_name: str = "grasp_system"
    level: str = "INFO"
    log_file: str = "logs/grasp_system.log"


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
