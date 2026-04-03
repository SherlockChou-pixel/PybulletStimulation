import argparse

from grasp_system.experiment_config import (
    add_behavior_args,
    add_environment_args,
    build_app_config_from_args,
)


def parse_args():
    parser = argparse.ArgumentParser(description="运行单次 GUI/DIRECT 多模态抓取实验。")
    parser.add_argument("--seed", type=int, default=1000, help="随机种子")
    parser.add_argument("--log-file", type=str, default="logs/grasp_system.log", help="日志文件路径")
    parser.add_argument("--direct", action="store_true", help="使用 DIRECT 模式而不是 GUI")
    add_environment_args(parser)
    add_behavior_args(parser)
    return parser.parse_args()


def build_config(args):
    return build_app_config_from_args(
        args,
        args.seed,
        logger_name="grasp_system",
        log_file=args.log_file,
        realtime_sleep=not args.direct,
        sleep_dt=0.0 if args.direct else (1 / 480.0),
    )


def print_run_summary(config, gui: bool):
    scene = config.scene
    print("=== 当前实验配置 ===")
    print(f"mode: {'GUI' if gui else 'DIRECT'}")
    print(f"scenario: {scene.scenario_name}")
    print(f"seed: {scene.random_seed}")
    print(f"object_urdf: {scene.object_urdf}")
    if scene.randomize_object_position:
        print(f"object_x_range: {scene.object_x_range}")
        print(f"object_y_range: {scene.object_y_range}")
        print(f"object_spawn_z: {scene.object_spawn_z}")
    else:
        print(f"fixed_object_position: {scene.object_position}")
    print(f"randomize_object_yaw: {scene.randomize_object_yaw}")
    print(f"object_yaw_range: {scene.object_yaw_range}")
    print(f"object_mass: {scene.object_mass}")
    print(f"object_lateral_friction: {scene.object_lateral_friction}")
    print(f"floor_lateral_friction: {scene.floor_lateral_friction}")
    print(f"closed_loop_enabled: {config.fusion.closed_loop_enabled}")
    print(f"recenter_enabled: {config.fusion.recenter_enabled}")
    print(f"place_enabled: {config.place.enabled}")
    print("====================")


def main():
    args = parse_args()
    try:
        from grasp_system import GraspWorkflow
    except ModuleNotFoundError as exc:
        if exc.name == "pybullet":
            raise SystemExit("缺少依赖 pybullet，请先安装后再运行 GUI 实验。") from exc
        raise

    config = build_config(args)
    gui = not args.direct
    print_run_summary(config, gui=gui)

    workflow = GraspWorkflow(config=config)
    result = workflow.run(gui=gui)
    print("=== 运行结果 ===")
    print(result)


if __name__ == "__main__":
    main()
