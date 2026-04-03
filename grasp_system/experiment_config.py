import argparse
from dataclasses import replace
from typing import Any, Optional

from .config import AppConfig


SCENARIO_PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {
        "scenario_name": "baseline",
    },
    "edge_reach": {
        "scenario_name": "edge_reach",
        "object_x_range": (0.66, 0.74),
        "object_y_range": (-0.34, 0.30),
        "randomize_object_yaw": True,
    },
    "low_friction": {
        "scenario_name": "low_friction",
        "object_lateral_friction": 0.22,
        "object_spinning_friction": 0.001,
        "object_rolling_friction": 0.001,
        "floor_lateral_friction": 0.35,
    },
    "heavy_object": {
        "scenario_name": "heavy_object",
        "object_mass": 0.35,
        "randomize_object_yaw": True,
    },
    "combined_challenge": {
        "scenario_name": "combined_challenge",
        "object_x_range": (0.66, 0.74),
        "object_y_range": (-0.34, 0.30),
        "object_mass": 0.35,
        "object_lateral_friction": 0.22,
        "object_spinning_friction": 0.001,
        "object_rolling_friction": 0.001,
        "floor_lateral_friction": 0.35,
        "randomize_object_yaw": True,
    },
}


def positive_float(value: str):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def nonnegative_float(value: str):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def add_environment_args(parser: argparse.ArgumentParser):
    parser.add_argument("--object-urdf", type=str, default="cube_small.urdf", help="实验物体 URDF")
    parser.add_argument(
        "--scenario",
        type=str,
        default="baseline",
        choices=tuple(SCENARIO_PRESETS.keys()),
        help="复杂环境预设场景",
    )
    parser.add_argument("--scenario-tag", type=str, default=None, help="覆盖写入结果中的场景标签")
    parser.add_argument("--object-x-range", type=float, nargs=2, metavar=("MIN_X", "MAX_X"), help="覆盖物体随机 X 范围")
    parser.add_argument("--object-y-range", type=float, nargs=2, metavar=("MIN_Y", "MAX_Y"), help="覆盖物体随机 Y 范围")
    parser.add_argument("--object-spawn-z", type=float, help="覆盖物体生成高度")
    parser.add_argument("--fixed-object-position", type=float, nargs=3, metavar=("X", "Y", "Z"), help="固定物体位置")
    parser.add_argument("--object-mass", type=positive_float, help="覆盖物体质量")
    parser.add_argument("--object-lateral-friction", type=positive_float, help="覆盖物体侧向摩擦系数")
    parser.add_argument("--object-spinning-friction", type=nonnegative_float, help="覆盖物体自旋摩擦")
    parser.add_argument("--object-rolling-friction", type=nonnegative_float, help="覆盖物体滚动摩擦")
    parser.add_argument("--floor-lateral-friction", type=positive_float, help="覆盖地面侧向摩擦系数")
    parser.add_argument("--object-restitution", type=nonnegative_float, help="覆盖物体弹性系数")
    parser.add_argument("--object-linear-damping", type=nonnegative_float, help="覆盖物体线性阻尼")
    parser.add_argument("--object-angular-damping", type=nonnegative_float, help="覆盖物体角阻尼")
    parser.add_argument("--randomize-object-yaw", action="store_true", help="启用物体初始偏航随机化")
    parser.add_argument("--object-yaw-range", type=float, nargs=2, metavar=("MIN_YAW", "MAX_YAW"), help="覆盖物体偏航随机范围（弧度）")
    return parser


def add_behavior_args(parser: argparse.ArgumentParser):
    parser.add_argument("--disable-place", action="store_true", help="关闭放置流程，仅测试抓取与抬升")
    parser.add_argument("--disable-closed-loop", action="store_true", help="关闭融合闭环夹持调节")
    parser.add_argument("--disable-recenter", action="store_true", help="关闭 recenter 抓取回正")
    parser.add_argument("--enable-refine-pass", action="store_true", help="启用二次视觉细化")
    parser.add_argument("--enable-auto-bias-calibration", action="store_true", help="启用仿真内视觉偏差自动校正")
    return parser


def _replace_if_present(scene, **kwargs):
    updates = {key: value for key, value in kwargs.items() if value is not None}
    if not updates:
        return scene
    return replace(scene, **updates)


def _normalize_range(values, label: str):
    if values is None:
        return None
    lo, hi = float(values[0]), float(values[1])
    if lo > hi:
        raise ValueError(f"{label} min must be <= max")
    return (lo, hi)


def build_scene_from_args(args, seed: Optional[int], base_scene):
    preset = SCENARIO_PRESETS[args.scenario]
    scenario_name = args.scenario_tag or preset.get("scenario_name") or args.scenario

    scene = replace(
        base_scene,
        random_seed=seed,
        object_urdf=args.object_urdf,
        **preset,
    )
    scene = replace(scene, scenario_name=scenario_name)

    x_range = _normalize_range(getattr(args, "object_x_range", None), "object_x_range")
    y_range = _normalize_range(getattr(args, "object_y_range", None), "object_y_range")
    yaw_range = _normalize_range(getattr(args, "object_yaw_range", None), "object_yaw_range")

    fixed_object_position = getattr(args, "fixed_object_position", None)
    if fixed_object_position is not None:
        fixed_position = tuple(float(value) for value in fixed_object_position)
        scene = replace(
            scene,
            randomize_object_position=False,
            object_position=fixed_position,
            object_spawn_z=float(fixed_position[2]),
        )
    else:
        scene = _replace_if_present(
            scene,
            object_x_range=x_range,
            object_y_range=y_range,
            object_spawn_z=float(args.object_spawn_z) if getattr(args, "object_spawn_z", None) is not None else None,
        )

    scene = _replace_if_present(
        scene,
        object_mass=getattr(args, "object_mass", None),
        object_lateral_friction=getattr(args, "object_lateral_friction", None),
        object_spinning_friction=getattr(args, "object_spinning_friction", None),
        object_rolling_friction=getattr(args, "object_rolling_friction", None),
        floor_lateral_friction=getattr(args, "floor_lateral_friction", None),
        object_restitution=getattr(args, "object_restitution", None),
        object_linear_damping=getattr(args, "object_linear_damping", None),
        object_angular_damping=getattr(args, "object_angular_damping", None),
        object_yaw_range=yaw_range,
    )

    if getattr(args, "randomize_object_yaw", False):
        scene = replace(scene, randomize_object_yaw=True)

    return scene


def build_app_config_from_args(args, seed: Optional[int], *, logger_name: str, log_file: Optional[str], realtime_sleep: bool, sleep_dt: float):
    base = AppConfig()
    runtime = replace(base.runtime, realtime_sleep=realtime_sleep, sleep_dt=sleep_dt)
    scene = build_scene_from_args(args, seed, base.scene)
    vision = replace(
        base.vision,
        enable_refine_pass=getattr(args, "enable_refine_pass", False),
        auto_bias_calibration_in_sim=getattr(args, "enable_auto_bias_calibration", False),
    )
    fusion = replace(
        base.fusion,
        closed_loop_enabled=not getattr(args, "disable_closed_loop", False),
        recenter_enabled=not getattr(args, "disable_recenter", False),
    )
    place = replace(base.place, enabled=not getattr(args, "disable_place", False))
    logging = replace(
        base.logging,
        logger_name=logger_name,
        log_file=log_file if log_file is not None else base.logging.log_file,
    )
    return replace(
        base,
        runtime=runtime,
        scene=scene,
        vision=vision,
        fusion=fusion,
        place=place,
        logging=logging,
    )


def scene_metadata(config: AppConfig):
    scene = config.scene
    return {
        "scenario_name": scene.scenario_name,
        "object_urdf": scene.object_urdf,
        "randomize_object_position": scene.randomize_object_position,
        "object_x_min": float(scene.object_x_range[0]),
        "object_x_max": float(scene.object_x_range[1]),
        "object_y_min": float(scene.object_y_range[0]),
        "object_y_max": float(scene.object_y_range[1]),
        "object_spawn_z_config": float(scene.object_spawn_z),
        "randomize_object_yaw": scene.randomize_object_yaw,
        "object_yaw_min": float(scene.object_yaw_range[0]),
        "object_yaw_max": float(scene.object_yaw_range[1]),
        "object_mass_config": scene.object_mass,
        "object_lateral_friction_config": scene.object_lateral_friction,
        "object_spinning_friction_config": scene.object_spinning_friction,
        "object_rolling_friction_config": scene.object_rolling_friction,
        "floor_lateral_friction_config": scene.floor_lateral_friction,
        "object_restitution_config": scene.object_restitution,
        "object_linear_damping_config": scene.object_linear_damping,
        "object_angular_damping_config": scene.object_angular_damping,
    }
