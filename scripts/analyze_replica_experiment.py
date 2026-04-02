from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ovo.utils import eval_utils


NUMERIC_LOG_KEYS = [
    "t_sam",
    "t_obj",
    "t_clip",
    "t_up",
    "t_seg",
    "t_lc",
    "n_matches",
    "n_obj",
    "vram",
    "vram_reserved",
    "vram_sam",
    "vram_obj",
    "vram_clip",
    "vram_up",
]


def read_log_file(path: Path) -> list[float]:
    if not path.exists():
        return []
    values: list[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            parsed = ast.literal_eval(line)
            if isinstance(parsed, list):
                values.extend(float(item) for item in parsed)
                continue
        values.append(float(line))
    return values


def summarize_series(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "max": np.nan,
            "sum": np.nan,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "sum": float(arr.sum()),
    }


def load_dataset_info(data_root: Path) -> dict[str, Any]:
    path = data_root / "working" / "configs" / "Replica" / "eval_info.yaml"
    return yaml.safe_load(path.read_text())


def compute_scene_metrics(pred_dir: Path, gt_dir: Path, dataset_info: dict[str, Any], scene: str) -> dict[str, Any]:
    metrics, _ = eval_utils.eval_semantics(
        str(pred_dir),
        str(gt_dir),
        [scene],
        dataset_info,
        verbose=False,
        return_metrics=True,
    )
    return metrics


def build_scene_summary(experiment_root: Path, dataset_info: dict[str, Any], gt_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pred_dir = experiment_root / dataset_info["dataset"]
    for scene in dataset_info["scenes"]:
        scene_root = experiment_root / scene
        logger_dir = scene_root / "logger"
        row: dict[str, Any] = {
            "scene": scene,
            "scene_root": str(scene_root),
            "has_scene_output": scene_root.exists(),
            "has_prediction": (pred_dir / f"{scene}.txt").exists(),
        }

        for key in NUMERIC_LOG_KEYS:
            stats = summarize_series(read_log_file(logger_dir / f"{key}.log"))
            for stat_name, value in stats.items():
                row[f"{key}_{stat_name}"] = value

        fps_stats = summarize_series(read_log_file(logger_dir / "avg_fps.log"))
        spf_stats = summarize_series(read_log_file(logger_dir / "spf.log"))
        row["avg_fps"] = fps_stats["mean"]
        row["spf_mean"] = spf_stats["mean"]
        row["spf_p95"] = spf_stats["p95"]
        row["max_vram"] = summarize_series(read_log_file(logger_dir / "max_vram.log"))["mean"]
        row["max_vram_reserved"] = summarize_series(read_log_file(logger_dir / "max_vram_reserved.log"))["mean"]
        row["max_ram"] = summarize_series(read_log_file(logger_dir / "max_ram.log"))["mean"]

        if row["has_prediction"]:
            metrics = compute_scene_metrics(pred_dir, gt_dir, dataset_info, scene)
            for key, value in metrics.items():
                row[key] = value
        else:
            for key in ["iou", "acc", "fiou", "facc", "iou_head", "acc_head", "iou_comm", "acc_comm", "iou_tail", "acc_tail"]:
                row[key] = np.nan

        module_sum = np.nansum([
            row["t_sam_sum"],
            row["t_obj_sum"],
            row["t_clip_sum"],
            row["t_up_sum"],
        ])
        row["module_time_sum"] = module_sum
        row["sam_share"] = row["t_sam_sum"] / module_sum if module_sum else np.nan
        row["obj_share"] = row["t_obj_sum"] / module_sum if module_sum else np.nan
        row["clip_share"] = row["t_clip_sum"] / module_sum if module_sum else np.nan
        row["up_share"] = row["t_up_sum"] / module_sum if module_sum else np.nan
        seg_total = row["t_seg_sum"]
        row["loop_closure_share"] = row["t_lc_sum"] / seg_total if seg_total else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values("scene").reset_index(drop=True)


def build_module_summary(scene_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    module_specs = [
        ("sam", "t_sam", "vram_sam"),
        ("object", "t_obj", "vram_obj"),
        ("clip", "t_clip", "vram_clip"),
        ("update", "t_up", "vram_up"),
        ("semantic_total", "t_seg", "vram"),
        ("loop_closure", "t_lc", "vram"),
    ]
    for module_name, time_key, mem_key in module_specs:
        rows.append(
            {
                "module": module_name,
                "mean_time_s": float(scene_df[f"{time_key}_mean"].mean()),
                "median_time_s": float(scene_df[f"{time_key}_median"].mean()),
                "p95_time_s": float(scene_df[f"{time_key}_p95"].mean()),
                "max_time_s": float(scene_df[f"{time_key}_max"].max()),
                "mean_vram_gb": float(scene_df[f"{mem_key}_mean"].mean()),
                "p95_vram_gb": float(scene_df[f"{mem_key}_p95"].mean()),
                "max_vram_gb": float(scene_df[f"{mem_key}_max"].max()),
            }
        )
    return pd.DataFrame(rows)


def build_report(scene_df: pd.DataFrame, module_df: pd.DataFrame) -> str:
    best_iou = scene_df.sort_values("iou", ascending=False).iloc[0]
    worst_iou = scene_df.sort_values("iou", ascending=True).iloc[0]
    slowest_scene = scene_df.sort_values("spf_mean", ascending=False).iloc[0]
    fastest_scene = scene_df.sort_values("spf_mean", ascending=True).iloc[0]
    highest_vram = scene_df.sort_values("max_vram_reserved", ascending=False).iloc[0]

    bottleneck = module_df.sort_values("mean_time_s", ascending=False).iloc[0]

    lines = [
        "# Replica Experiment Analysis",
        "",
        "## Overview",
        f"- Scenes analyzed: {len(scene_df)}",
        f"- Mean scene mIoU: {scene_df['iou'].mean():.3f}",
        f"- Mean scene mAcc: {scene_df['acc'].mean():.3f}",
        f"- Mean scene FPS: {scene_df['avg_fps'].mean():.3f}",
        f"- Highest average time module: {bottleneck['module']} ({bottleneck['mean_time_s']:.3f}s)",
        "",
        "## Scene Highlights",
        f"- Best mIoU: {best_iou['scene']} ({best_iou['iou']:.3f})",
        f"- Worst mIoU: {worst_iou['scene']} ({worst_iou['iou']:.3f})",
        f"- Fastest scene: {fastest_scene['scene']} ({fastest_scene['spf_mean']:.3f}s/step)",
        f"- Slowest scene: {slowest_scene['scene']} ({slowest_scene['spf_mean']:.3f}s/step)",
        f"- Highest reserved VRAM: {highest_vram['scene']} ({highest_vram['max_vram_reserved']:.3f} GB)",
        "",
        "## Bottleneck Notes",
    ]

    for _, row in scene_df.sort_values("spf_mean", ascending=False).iterrows():
        dominant = max(
            [
                ("SAM", row["sam_share"]),
                ("Object", row["obj_share"]),
                ("CLIP", row["clip_share"]),
                ("Update", row["up_share"]),
            ],
            key=lambda item: item[1] if not np.isnan(item[1]) else -1,
        )
        lines.append(
            f"- {row['scene']}: dominant module {dominant[0]} ({dominant[1]:.1%}), "
            f"mIoU {row['iou']:.3f}, mean step {row['spf_mean']:.3f}s, max reserved VRAM {row['max_vram_reserved']:.3f} GB"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--data-root", default="/ws/data/OVO")
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root)
    data_root = Path(args.data_root)
    analysis_dir = experiment_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = load_dataset_info(data_root)
    gt_dir = data_root / "input" / "Datasets" / "Replica" / "semantic_gt"

    scene_df = build_scene_summary(experiment_root, dataset_info, gt_dir)
    module_df = build_module_summary(scene_df)

    scene_df.to_csv(analysis_dir / "scene_summary.csv", index=False)
    module_df.to_csv(analysis_dir / "module_summary.csv", index=False)

    payload = {
        "experiment_root": str(experiment_root),
        "scene_summary": scene_df.to_dict(orient="records"),
        "module_summary": module_df.to_dict(orient="records"),
    }
    (analysis_dir / "analysis.json").write_text(json.dumps(payload, indent=2))
    (analysis_dir / "report.md").write_text(build_report(scene_df, module_df))

    print(f"Wrote analysis to {analysis_dir}")


if __name__ == "__main__":
    main()
