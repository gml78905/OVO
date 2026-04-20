from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ovo.utils import io_utils, path_utils
from ovo.utils.gt_mask_utils import precompute_replica_gt_masks


def load_scene_runtime_config(dataset_name: str, scene_name: str) -> dict:
    config = io_utils.load_config(path_utils.get_configs_root() / "ovo.yaml")
    config = path_utils.remap_data_paths(config)

    config_dataset = io_utils.load_config(path_utils.get_configs_root() / dataset_name / f"{dataset_name.lower()}.yaml")
    config_dataset = path_utils.remap_data_paths(config_dataset)
    io_utils.update_recursive(config, config_dataset)

    scene_config_path = path_utils.get_configs_root() / dataset_name / f"{scene_name}.yaml"
    if scene_config_path.exists():
        config_scene = io_utils.load_config(scene_config_path)
        config_scene = path_utils.remap_data_paths(config_scene)
        io_utils.update_recursive(config, config_scene)

    if "data" not in config:
        config["data"] = {}
    config["data"]["scene_name"] = scene_name
    config["data"]["input_path"] = str(path_utils.get_datasets_root() / dataset_name / scene_name)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Replica GT instance masks in OVO precomputed-mask format.")
    parser.add_argument("--scene", required=True, type=str)
    parser.add_argument("--dataset_name", default="Replica", type=str)
    parser.add_argument("--replica_root", required=True, type=Path)
    parser.add_argument("--output_dir", default=None, type=Path)
    parser.add_argument("--segment_every", default=None, type=int)
    parser.add_argument("--max_distance", default=0.05, type=float)
    parser.add_argument("--min_area", default=20, type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_scene_runtime_config(args.dataset_name, args.scene)
    output_dir = args.output_dir or (path_utils.get_working_root() / "masks" / "replica_gt_instance")
    segment_every = args.segment_every or config["semantic"].get("segment_every", 10)

    scene_output_dir = precompute_replica_gt_masks(
        {"dataset_name": config["dataset_name"], "data": {**config["data"], **config["cam"]}},
        args.scene,
        args.replica_root,
        output_dir,
        segment_every=segment_every,
        max_distance=args.max_distance,
        min_area=args.min_area,
        force=args.force,
    )
    print(f"Saved GT masks to {scene_output_dir}")


if __name__ == "__main__":
    main()
