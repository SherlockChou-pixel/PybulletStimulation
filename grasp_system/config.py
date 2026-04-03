from dataclasses import dataclass, field
from typing import Optional

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
    top_percentile: float = 98.0
    center_low_percentile: float = 5.0
    center_high_percentile: float = 95.0
    top_surface_band: float = 0.003
    enable_refine_pass: bool = False
    auto_bias_calibration_in_sim: bool = False
    bias_update_alpha: float = 0.5
    replan_position_threshold: float = 0.01
    refine_accept_max_shift: float = 0.008


@dataclass(frozen=True)
class SceneConfig:
    scenario_name: str = "baseline"
    plane_urdf: str = "plane.urdf"
    robot_urdf: str = "franka_panda/panda.urdf"
    object_urdf: str = "cube_small.urdf"
    robot_base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    object_position: tuple[float, float, float] = (0.6, 0.0, 0.2)
    randomize_object_position: bool = True
    object_x_range: tuple[float, float] = (0.42, 0.72)
    object_y_range: tuple[float, float] = (-0.30, 0.25)
    object_spawn_z: float = 0.20
    object_orientation_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    randomize_object_yaw: bool = False
    object_yaw_range: tuple[float, float] = (-np.pi, np.pi)
    object_mass: Optional[float] = None
    object_lateral_friction: Optional[float] = None
    object_spinning_friction: Optional[float] = None
    object_rolling_friction: Optional[float] = None
    object_restitution: Optional[float] = None
    object_linear_damping: Optional[float] = None
    object_angular_damping: Optional[float] = None
    floor_lateral_friction: Optional[float] = None
    floor_spinning_friction: Optional[float] = None
    floor_rolling_friction: Optional[float] = None
    random_seed: Optional[int] = None
    use_fixed_base: bool = True
    cam_to_world_bias: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))


@dataclass(frozen=True)
class RuntimeConfig:
    sim_dt: float = 1 / 240.0
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
    carry_force_multiplier: float = 1.5
    carry_force_cap: float = 30.0


@dataclass(frozen=True)
class IMUConfig:
    enabled: bool = True
    history_length: int = 240
    summary_window: int = 40
    sample_every_steps: int = 4
    track_end_effector: bool = True
    track_object: bool = True
    log_samples: bool = False


@dataclass(frozen=True)
class PressureConfig:
    enabled: bool = True
    history_length: int = 240
    summary_window: int = 40
    sample_every_steps: int = 4
    min_active_fingers: int = 2
    min_total_normal_force: float = 5.0
    log_samples: bool = False


@dataclass(frozen=True)
class FusionConfig:
    enabled: bool = True
    vision_good_error: float = 0.005
    vision_max_error: float = 0.03
    desired_total_normal_force: float = 24.0
    force_balance_tolerance: float = 0.35
    lateral_ratio_warn: float = 0.45
    object_speed_stable: float = 0.02
    object_peak_acc_stable: float = 1.5
    end_effector_peak_acc_smooth: float = 8.0
    slip_object_speed_warn: float = 0.03
    pressure_drop_warn_ratio: float = 0.25
    stable_grasp_confidence: float = 0.65
    high_slip_risk: float = 0.55
    closed_loop_enabled: bool = True
    control_monitor_interval: int = 12
    control_cooldown_steps: int = 36
    control_slip_risk: float = 0.22
    control_low_pressure_confidence: float = 0.78
    control_low_grasp_confidence: float = 0.72
    control_force_step: float = 2.0
    control_target_step: float = 0.0008
    control_force_cap: float = 30.0
    control_force_hold_threshold: float = 22.0
    control_motion_stable_threshold: float = 0.010
    control_max_actions_per_stage: int = 6
    control_transport_slip_risk: float = 0.30
    control_transport_low_grasp_confidence: float = 0.60
    control_transport_pressure_confidence: float = 0.90
    control_transport_lateral_ratio: float = 0.40
    control_transport_object_speed: float = 0.015
    control_place_stable_slip_risk: float = 0.28
    control_place_stable_grasp_confidence: float = 0.72
    control_place_stable_pressure_confidence: float = 0.88
    control_place_stable_lateral_ratio: float = 0.45
    control_place_max_actions_stable: int = 1
    post_recenter_grace_steps: int = 72
    transport_motion_penalty_scale: float = 0.45
    transport_stable_grasp_boost: float = 0.08
    recenter_enabled: bool = True
    recenter_lateral_ratio_trigger: float = 0.52
    recenter_object_xy_trigger: float = 0.0025
    recenter_xy_gain: float = 0.85
    recenter_max_step: float = 0.006
    recenter_max_attempts: int = 2
    validation_min_grasp_confidence: float = 0.78
    validation_max_slip_risk: float = 0.32
    validation_max_lateral_ratio: float = 0.58


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
        -5 * np.pi / 6,
        -3 * np.pi / 4,
        -2 * np.pi / 3,
        -np.pi / 2,
        -np.pi / 4,
        -np.pi / 6,
        0.0,
        np.pi / 6,
        np.pi / 4,
        np.pi / 2,
        2 * np.pi / 3,
        3 * np.pi / 4,
        5 * np.pi / 6,
        np.pi,
    )
    grasp_tilt_candidates_deg: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 12.0),
        (0.0, -12.0),
        (12.0, 0.0),
        (-12.0, 0.0),
        (8.0, 8.0),
        (8.0, -8.0),
        (-8.0, 8.0),
        (-8.0, -8.0),
        (18.0, 0.0),
        (-18.0, 0.0),
    )
    grasp_xy_search_step: float = 0.015
    grasp_xy_search_radius: float = 0.04
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
    recenter_steps: int = 120
    recenter_settle_steps: int = 24
    max_grasp_candidate_trials: int = 5


@dataclass(frozen=True)
class PlaceConfig:
    enabled: bool = True
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
    imu: IMUConfig = field(default_factory=IMUConfig)
    pressure: PressureConfig = field(default_factory=PressureConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    place: PlaceConfig = field(default_factory=PlaceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
