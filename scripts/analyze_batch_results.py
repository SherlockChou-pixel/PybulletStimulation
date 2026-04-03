import argparse
import csv
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="分析批量实验 CSV，并生成汇总与图表。")
    parser.add_argument("--input", type=str, default="logs/batch_experiments.csv", help="输入 CSV 文件")
    parser.add_argument("--outdir", type=str, default="logs/batch_analysis", help="分析输出目录")
    return parser.parse_args()


def as_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value):
    return str(value).strip().lower() == "true"


def load_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def metric_values(rows, key):
    values = [as_float(row.get(key)) for row in rows]
    return [value for value in values if value is not None]


def safe_mean(values):
    return statistics.mean(values) if values else None


def correlation(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    den = math.sqrt(den_x * den_y)
    if den == 0:
        return None
    return num / den


def summarize(rows):
    total = len(rows)
    status_counts = Counter(row.get("status", "") for row in rows)
    grasp_success = sum(as_bool(row.get("grasp_success")) for row in rows)
    workflow_finished = sum(as_bool(row.get("workflow_finished")) for row in rows)
    place_attempted = sum(as_bool(row.get("place_attempted")) for row in rows)
    place_success = sum(as_bool(row.get("place_success")) for row in rows)
    retries = 0
    for row in rows:
        if "had_retry" in row and row.get("had_retry") not in (None, ""):
            retries += 1 if as_bool(row.get("had_retry")) else 0
        else:
            attempts_used = as_float(row.get("attempts_used")) or 0.0
            retries += 1 if attempts_used > 1.0 else 0

    summary = {
        "total_runs": total,
        "status_counts": dict(status_counts),
        "grasp_success_rate": grasp_success / total if total else None,
        "workflow_finished_rate": workflow_finished / total if total else None,
        "place_success_rate": place_success / place_attempted if place_attempted else None,
        "retry_rate": retries / total if total else None,
    }

    numeric_keys = [
        "vision_error",
        "attempts_used",
        "fusion_control_count",
        "recenter_count",
        "sim_time",
        "after_lift_grasp_confidence",
        "after_lift_slip_risk",
        "before_release_grasp_confidence",
        "before_release_slip_risk",
    ]
    for key in numeric_keys:
        values = metric_values(rows, key)
        if values:
            summary[key] = {
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
            }

    summary["failure_reasons"] = dict(
        Counter(
            row.get("failure_reason", "")
            for row in rows
            if not as_bool(row.get("workflow_finished")) and row.get("failure_reason")
        )
    )

    after_lift_conf = [as_float(row.get("after_lift_grasp_confidence")) for row in rows]
    fusion_counts = [as_float(row.get("fusion_control_count")) for row in rows]
    retry_counts = [as_float(row.get("attempts_used")) for row in rows]
    summary["corr_after_lift_conf_vs_fusion_control"] = correlation(after_lift_conf, fusion_counts)
    summary["corr_after_lift_conf_vs_attempts_used"] = correlation(after_lift_conf, retry_counts)

    return summary


def build_markdown_summary(summary):
    lines = [
        "# Batch Experiment Summary",
        "",
        f"- Total runs: {summary['total_runs']}",
        f"- Status counts: {summary['status_counts']}",
        f"- Grasp success rate: {summary['grasp_success_rate']:.2%}" if summary.get("grasp_success_rate") is not None else "- Grasp success rate: N/A",
        f"- Workflow finished rate: {summary['workflow_finished_rate']:.2%}" if summary.get("workflow_finished_rate") is not None else "- Workflow finished rate: N/A",
        f"- Place success rate: {summary['place_success_rate']:.2%}" if summary.get("place_success_rate") is not None else "- Place success rate: N/A",
        f"- Retry rate: {summary['retry_rate']:.2%}" if summary.get("retry_rate") is not None else "- Retry rate: N/A",
        "",
        "## Key metrics",
        "",
    ]

    metric_labels = {
        "vision_error": "Vision error",
        "attempts_used": "Attempts used",
        "fusion_control_count": "Fusion control count",
        "recenter_count": "Recenter count",
        "sim_time": "Simulation time",
        "after_lift_grasp_confidence": "After-lift grasp confidence",
        "after_lift_slip_risk": "After-lift slip risk",
        "before_release_grasp_confidence": "Before-release grasp confidence",
        "before_release_slip_risk": "Before-release slip risk",
    }
    for key, label in metric_labels.items():
        metric = summary.get(key)
        if metric:
            lines.append(
                f"- {label}: mean={metric['mean']:.4f}, min={metric['min']:.4f}, max={metric['max']:.4f}"
            )

    lines.extend(
        [
            "",
            "## Correlations",
            "",
            f"- corr(after_lift_grasp_confidence, fusion_control_count) = {summary['corr_after_lift_conf_vs_fusion_control']:.4f}"
            if summary.get("corr_after_lift_conf_vs_fusion_control") is not None
            else "- corr(after_lift_grasp_confidence, fusion_control_count) = N/A",
            f"- corr(after_lift_grasp_confidence, attempts_used) = {summary['corr_after_lift_conf_vs_attempts_used']:.4f}"
            if summary.get("corr_after_lift_conf_vs_attempts_used") is not None
            else "- corr(after_lift_grasp_confidence, attempts_used) = N/A",
            "",
            "## Failure reasons",
            "",
            f"- {summary['failure_reasons']}" if summary.get("failure_reasons") else "- No final failures in this batch.",
            "",
            "## Quick observations",
            "",
        ]
    )

    observations = []
    success_rate = summary.get("workflow_finished_rate")
    if success_rate is not None:
        if success_rate >= 0.95:
            observations.append("Final success rate is high, so the full workflow is already stable in this batch.")
        elif success_rate >= 0.80:
            observations.append("Success rate is acceptable, but there is still room to improve robustness.")
        else:
            observations.append("Success rate is low; priority should be improving the base grasp pipeline before detailed ablations.")

    retry_rate = summary.get("retry_rate")
    attempts_metric = summary.get("attempts_used")
    if retry_rate is not None and attempts_metric:
        if retry_rate > 0.30 or attempts_metric["mean"] > 1.3:
            observations.append("A noticeable fraction of runs needed multiple grasp attempts, so initial pose selection is not always optimal.")
        else:
            observations.append("Most runs succeed on the first candidate, indicating that pose planning is already reliable.")

    fusion_metric = summary.get("fusion_control_count")
    if fusion_metric:
        if fusion_metric["mean"] >= 6:
            observations.append("Closed-loop control intervenes frequently; current robustness depends strongly on online correction.")
        else:
            observations.append("Closed-loop interventions are relatively limited, suggesting the nominal grasp is already stable.")

    lift_slip = summary.get("after_lift_slip_risk")
    release_slip = summary.get("before_release_slip_risk")
    if lift_slip and release_slip:
        if release_slip["mean"] > lift_slip["mean"] + 0.05:
            observations.append("Slip risk becomes higher before release than right after lift, so transport/place stages introduce extra disturbance.")
        else:
            observations.append("Slip risk remains relatively consistent from lift to release.")

    for item in observations:
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def create_plots(rows, outdir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_idx = [int(row["run_index"]) for row in rows]
    grasp_success = [1 if as_bool(row.get("grasp_success")) else 0 for row in rows]
    workflow_finished = [1 if as_bool(row.get("workflow_finished")) else 0 for row in rows]
    place_success = [1 if as_bool(row.get("place_success")) else 0 for row in rows]
    vision_error = [as_float(row.get("vision_error")) or 0.0 for row in rows]
    attempts_used = [as_float(row.get("attempts_used")) or 0.0 for row in rows]
    fusion_control = [as_float(row.get("fusion_control_count")) or 0.0 for row in rows]
    recenter_count = [as_float(row.get("recenter_count")) or 0.0 for row in rows]
    after_lift_conf = [as_float(row.get("after_lift_grasp_confidence")) or 0.0 for row in rows]
    after_lift_slip = [as_float(row.get("after_lift_slip_risk")) or 0.0 for row in rows]
    before_release_slip = [as_float(row.get("before_release_slip_risk")) or 0.0 for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].bar(["grasp", "workflow", "place"], [sum(grasp_success), sum(workflow_finished), sum(place_success)], color=["#4caf50", "#2196f3", "#ff9800"])
    axes[0, 0].set_title("Success counts")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].hist(vision_error, bins=min(8, max(3, len(rows) // 2)), color="#607d8b", edgecolor="black")
    axes[0, 1].set_title("Vision error distribution")
    axes[0, 1].set_xlabel("vision_error")
    axes[0, 1].set_ylabel("count")

    axes[1, 0].plot(run_idx, fusion_control, marker="o", label="fusion_control_count")
    axes[1, 0].plot(run_idx, recenter_count, marker="s", label="recenter_count")
    axes[1, 0].plot(run_idx, attempts_used, marker="^", label="attempts_used")
    axes[1, 0].set_title("Control actions by run")
    axes[1, 0].set_xlabel("run_index")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    scatter = axes[1, 1].scatter(after_lift_conf, after_lift_slip, c=attempts_used, cmap="viridis", s=70)
    axes[1, 1].set_title("After-lift confidence vs slip risk")
    axes[1, 1].set_xlabel("after_lift_grasp_confidence")
    axes[1, 1].set_ylabel("after_lift_slip_risk")
    axes[1, 1].grid(alpha=0.3)
    cbar = fig.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label("attempts_used")

    fig.tight_layout()
    fig.savefig(outdir / "batch_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(run_idx, after_lift_conf, marker="o", label="after_lift_grasp_confidence", color="#2e7d32")
    axes[0].plot(run_idx, [as_float(row.get("before_release_grasp_confidence")) or 0.0 for row in rows], marker="s", label="before_release_grasp_confidence", color="#1565c0")
    axes[0].set_ylabel("confidence")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Confidence trends by run")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(run_idx, after_lift_slip, marker="o", label="after_lift_slip_risk", color="#ef6c00")
    axes[1].plot(run_idx, before_release_slip, marker="s", label="before_release_slip_risk", color="#8e24aa")
    axes[1].set_xlabel("run_index")
    axes[1].set_ylabel("slip_risk")
    axes[1].set_ylim(0.0, max(before_release_slip + after_lift_slip + [0.35]) * 1.1)
    axes[1].set_title("Slip-risk trends by run")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "quality_trends.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"找不到输入文件：{input_path}")

    rows = load_rows(input_path)
    if not rows:
        raise SystemExit("CSV 为空，无法分析。")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = summarize(rows)
    summary_md = build_markdown_summary(summary)
    (outdir / "summary.md").write_text(summary_md, encoding="utf-8")
    create_plots(rows, outdir)

    print(summary_md)
    print(f"图表已输出到：{outdir}")


if __name__ == "__main__":
    main()
