from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ovo.utils.instance_eval import (
    evaluate_ap50,
    load_replica_faces,
    load_replica_gt_instances,
    load_replica_pred_instances,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Replica instance segmentation AP50")
    parser.add_argument("--pred_dir", required=True, type=Path)
    parser.add_argument("--replica_root", required=True, type=Path)
    parser.add_argument("--eval_info", required=True, type=Path)
    parser.add_argument("--scenes", nargs="+", required=True)
    args = parser.parse_args()

    eval_info = yaml.full_load(args.eval_info.read_text())
    map_to_reduced = eval_info["map_to_reduced"]
    class_names = eval_info["class_names_reduced"]

    all_predictions = []
    all_ground_truths = []
    for scene in args.scenes:
        replica_scene_dir = args.replica_root / scene.replace("office", "office_").replace("room", "room_")
        faces = load_replica_faces(replica_scene_dir)
        all_predictions.extend(load_replica_pred_instances(args.pred_dir, scene, faces))
        all_ground_truths.extend(load_replica_gt_instances(replica_scene_dir, map_to_reduced))

    per_class_ap, map50 = evaluate_ap50(all_predictions, all_ground_truths, iou_threshold=0.5)
    print(f"mAP50: {map50:.4f}")
    for class_id in sorted(per_class_ap):
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        print(f"{class_name}: {per_class_ap[class_id]:.4f}")


if __name__ == "__main__":
    main()
