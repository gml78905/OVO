from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ovo.utils.instance_eval import (
    evaluate_ap,
    evaluate_instance_miou,
    load_replica_faces,
    load_replica_gt_instances,
    load_replica_pred_instances,
)


def scene_to_replica_dir(replica_root: Path, scene: str) -> Path:
    if scene.startswith("office"):
        return replica_root / scene.replace("office", "office_")
    if scene.startswith("room"):
        return replica_root / scene.replace("room", "room_")
    return replica_root / scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Replica instance metrics (AP25/AP50/AP75/mIoU)")
    parser.add_argument("--pred_dir", required=True, type=Path)
    parser.add_argument("--replica_root", required=True, type=Path)
    parser.add_argument("--eval_info", required=True, type=Path)
    parser.add_argument("--scenes", nargs="+", required=True)
    args = parser.parse_args()

    eval_info = yaml.full_load(args.eval_info.read_text())
    map_to_reduced = eval_info["map_to_reduced"]

    all_predictions = []
    all_ground_truths = []
    used_scenes: List[str] = []
    skipped_scenes: List[str] = []

    for scene in args.scenes:
        replica_scene_dir = scene_to_replica_dir(args.replica_root, scene)
        mesh_path = replica_scene_dir / "habitat" / "mesh_semantic.ply"
        info_path = replica_scene_dir / "habitat" / "info_semantic.json"
        pred_file = args.pred_dir / f"{scene}.txt"
        if not mesh_path.exists() or not info_path.exists() or not pred_file.exists():
            skipped_scenes.append(scene)
            continue

        faces = load_replica_faces(replica_scene_dir)
        all_predictions.extend(load_replica_pred_instances(args.pred_dir, scene, faces))
        all_ground_truths.extend(load_replica_gt_instances(replica_scene_dir, map_to_reduced))
        used_scenes.append(scene)

    _, ap25 = evaluate_ap(all_predictions, all_ground_truths, iou_threshold=0.25)
    _, ap50 = evaluate_ap(all_predictions, all_ground_truths, iou_threshold=0.50)
    _, ap75 = evaluate_ap(all_predictions, all_ground_truths, iou_threshold=0.75)
    miou = evaluate_instance_miou(all_predictions, all_ground_truths)

    print(f"Scenes used: {', '.join(used_scenes) if used_scenes else '(none)'}")
    if skipped_scenes:
        print(f"Scenes skipped (missing GT or predictions): {', '.join(skipped_scenes)}")
    print(f"AP25: {ap25:.4f}")
    print(f"AP50: {ap50:.4f}")
    print(f"AP75: {ap75:.4f}")
    print(f"mIoU: {miou:.4f}")
    print(f"GT instances evaluated: {len(all_ground_truths)}")
    print(f"Pred instances evaluated: {len(all_predictions)}")


if __name__ == "__main__":
    main()
