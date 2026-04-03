import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_system.experiment_config import (
    add_behavior_args,
    add_environment_args,
    build_app_config_from_args,
    scene_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(description="批量运行多模态抓取仿真实验并导出 CSV 结果。")
    parser.add_argument("--runs", type=int, default=10, help="实验运行次数")
    parser.add_argument("--output", type=str, default="logs/batch_experiments.csv", help="CSV 输出路径")
    parser.add_argument("--log-file", type=str, default="logs/grasp_system_batch.log", help="批量实验日志文件")
    parser.add_argument("--base-seed", type=int, default=1000, help="起始随机种子，按 base_seed + run_index 递增")
    parser.add_argument("--gui", action="store_true", help="是否启用 PyBullet GUI")
    add_environment_args(parser)
    add_behavior_args(parser)
    return parser.parse_args()


def build_config(args, seed: int):
    return build_app_config_from_args(
        args,
        seed,
        logger_name="grasp_system_batch",
        log_file=args.log_file,
        realtime_sleep=False,
        sleep_dt=0.0,
    )


def normalize_row(run_index: int, seed: int, config, result: dict):
    row = {
        "run_index": run_index,
        "seed": seed,
        **scene_metadata(config),
    }
    row.update(result)
    return row


def build_exception_row(run_index: int, seed: int, config, exc: Exception):
    row = normalize_row(
        run_index,
        seed,
        config,
        {
            "status": "exception",
            "workflow_finished": False,
            "grasp_success": False,
            "place_attempted": False,
            "place_success": False,
            "failure_reason": str(exc),
            "exception_type": exc.__class__.__name__,
        },
    )
    return row


def write_rows(output_path: Path, rows: list[dict]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict], output_path: Path):
    total = len(rows)
    grasp_success = sum(1 for row in rows if row.get("grasp_success"))
    workflow_finished = sum(1 for row in rows if row.get("workflow_finished"))
    place_attempted = sum(1 for row in rows if row.get("place_attempted"))
    place_success = sum(1 for row in rows if row.get("place_success"))
    recenter_count = sum(int(row.get("recenter_count") or 0) for row in rows)
    fusion_control_count = sum(int(row.get("fusion_control_count") or 0) for row in rows)
    scenarios = sorted({str(row.get("scenario_name", "unknown")) for row in rows})

    print(f"批量实验完成：{total} 次")
    print(f"场景：{', '.join(scenarios)}")
    print(f"抓取成功：{grasp_success}/{total}")
    print(f"完整流程完成：{workflow_finished}/{total}")
    if place_attempted:
        print(f"放置成功：{place_success}/{place_attempted}")
    print(f"recenter 总触发次数：{recenter_count}")
    print(f"闭环控制总触发次数：{fusion_control_count}")
    print(f"CSV 已保存到：{output_path}")


def main():
    args = parse_args()
    try:
        from grasp_system import GraspWorkflow
    except ModuleNotFoundError as exc:
        if exc.name == "pybullet":
            raise SystemExit("缺少依赖 pybullet，请先安装后再运行批量实验脚本。") from exc
        raise

    rows = []

    for run_index in range(1, args.runs + 1):
        seed = args.base_seed + run_index - 1
        print(f"[Run {run_index}/{args.runs}] seed={seed} scenario={args.scenario_tag or args.scenario}")
        config = build_config(args, seed)
        workflow = GraspWorkflow(config=config)
        try:
            result = workflow.run(gui=args.gui)
            if result is None:
                result = {
                    "status": "unknown",
                    "workflow_finished": False,
                    "grasp_success": False,
                    "place_attempted": False,
                    "place_success": False,
                    "failure_reason": "workflow_returned_none",
                }
            rows.append(normalize_row(run_index, seed, config, result))
        except Exception as exc:
            rows.append(build_exception_row(run_index, seed, config, exc))
            print(f"  -> 异常：{exc.__class__.__name__}: {exc}")

    output_path = Path(args.output)
    write_rows(output_path, rows)
    print_summary(rows, output_path)


if __name__ == "__main__":
    main()
