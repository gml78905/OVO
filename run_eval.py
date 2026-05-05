from typing import Dict
from datetime import datetime
from pathlib import Path
import argparse
import wandb
import torch
import numpy as np
import time
import yaml
import uuid
import gc
import os
import shutil

# SAM2: transformer reads get_sdpa_settings() at import time — patch before any sam2 submodule import.
if os.environ.get("OVO_SAM2_ALLOW_FLASH", "").lower() not in ("1", "true", "yes"):
    try:
        import sam2.utils.misc as _sam2_misc

        _sam2_misc.get_sdpa_settings = lambda: (False, False, True)
    except ImportError:
        pass

from ovo.utils import io_utils, gen_utils, eval_utils, path_utils
from ovo.utils.gt_mask_utils import precompute_replica_gt_masks
from ovo.entities.ovomapping import OVOSemMap
from ovo.entities.ovo import OVO

def load_representation(scene_path: Path, eval: bool=False, debug_info: bool=False) -> OVO:
    config = io_utils.load_config(scene_path / "config.yaml", inherit=False)
    submap_ckpt = torch.load(scene_path /"ovo_map.ckpt" )
    map_params = submap_ckpt.get("map_params", None)
    if map_params is None:
        map_params = submap_ckpt["gaussian_params"]        
    config["semantic"]["verbose"] = False 
    ovo = OVO(config["semantic"],None, config["data"]["scene_name"], eval=eval, device=config.get("device", "cuda"))
    ovo.restore_dict(submap_ckpt["ovo_map_params"], debug_info=debug_info)
    return ovo, map_params


def compute_scene_labels(scene_path: Path, dataset_name: str, scene_name: str, data_path:str, dataset_info: Dict) -> None:

    ovo, map_params = load_representation(scene_path, eval=True)
    pcd_pred = map_params["xyz"]
    points_obj_ids = map_params["obj_ids"]

    _, pcd_gt = io_utils.load_scene_data(dataset_name, scene_name, data_path, dataset_info, False)
    classes = dataset_info["class_names"] if dataset_info.get("map_to_reduced", None) is None else dataset_info["class_names_reduced"]
    pred_path = scene_path.parent / dataset_info["dataset"]
    os.makedirs(pred_path, exist_ok=True)
    pred_path = pred_path / (scene_name+".txt")

    # It may happen that all the points associated to an object where prunned, such that the number of unique labels in points_obj_ids, is different from the number of semantic module instances
    print("Computing predicted instances labels ...")

    instances_info = ovo.classify_instances(classes)

    mesh_semantic_labels = dict()
    print("Matching instances to ground truth mesh ...")
    mesh_instance_labels, mesh_instances_masks, matched_instances_ids = eval_utils.match_labels_to_vtx(points_obj_ids[:,0], pcd_pred, pcd_gt)
    
    map_id_to_idx = {id: i for i, id in enumerate(ovo.objects.keys())}
    mesh_semantic_labels = instances_info["classes"][np.vectorize(map_id_to_idx.get)(mesh_instance_labels)]
    instances_info["masks"] = mesh_instances_masks.int().numpy()

    print(f"Writing prediction to {pred_path}!")
    io_utils.write_labels(pred_path, mesh_semantic_labels)
    io_utils.write_instances(scene_path.parent, scene_name, instances_info)

    ovo.cpu()
    del ovo


def run_scene(
    scene: str,
    dataset: str,
    experiment_name: str,
    tmp_run: bool = False,
    depth_filter: bool = None,
    use_gt_masks: bool = False,
    replica_gt_root: str | None = None,
    save_live_instance_vis: bool = False,
    save_selective_prompt_debug: bool = False,
    use_selective_prompt_points: bool = False,
) -> None:

    config = io_utils.load_config(path_utils.get_configs_root() / "ovo.yaml")
    config = path_utils.remap_data_paths(config)
    map_module = config["slam"]["slam_module"]
    if map_module.startswith("orbslam"):
        map_module = "vanilla"
        
    config_slam = io_utils.load_config(Path(config["slam"]["config_path"]) / map_module / f"{dataset.lower()}.yaml")
    config_slam = path_utils.remap_data_paths(config_slam)
    io_utils.update_recursive(config, config_slam)

    config_dataset = io_utils.load_config(path_utils.get_configs_root() / dataset / f"{dataset.lower()}.yaml")
    config_dataset = path_utils.remap_data_paths(config_dataset)
    io_utils.update_recursive(config, config_dataset)
    
    scene_config_path = path_utils.get_configs_root() / dataset / f"{scene}.yaml"
    if scene_config_path.exists():
        config_scene = io_utils.load_config(scene_config_path)
        config_scene = path_utils.remap_data_paths(config_scene)
        io_utils.update_recursive(config, config_scene)
        
    if "data" not in config:
        config["data"] = {}
    config["data"]["scene_name"] = scene
    config["data"]["input_path"] = str(path_utils.get_datasets_root() / dataset / scene)

    output_path = path_utils.get_output_root() / dataset

    if tmp_run:
        output_path = output_path / "tmp"

    output_path = output_path / experiment_name / scene

    if depth_filter is not None:
        config["semantic"]["depth_filter"] = depth_filter

    if save_live_instance_vis:
        config["semantic"].setdefault("live_instance_vis", {})
        config["semantic"]["live_instance_vis"]["enabled"] = True
    if save_selective_prompt_debug:
        config["semantic"].setdefault("selective_prompt_debug", {})
        config["semantic"]["selective_prompt_debug"]["enabled"] = True
    if use_selective_prompt_points:
        config["semantic"]["sam"]["use_prompt_plan_points"] = True

    if use_gt_masks or config["semantic"].get("use_gt_masks", False):
        if dataset.lower() != "replica":
            raise ValueError("GT instance masks are currently only supported for Replica scenes.")
        gt_masks_root = path_utils.get_working_root() / "masks" / "replica_gt_instance"
        gt_root = replica_gt_root or os.environ.get("OVO_REPLICA_GT_ROOT", "/ws/data/replica_v1")
        config["semantic"]["sam"]["precomputed"] = True
        config["semantic"]["sam"]["precompute"] = False
        config["semantic"]["sam"]["masks_base_path"] = str(gt_masks_root)
        print(f"Using Replica GT instance masks from {gt_root}")
        precompute_replica_gt_masks(
            {"dataset_name": config["dataset_name"], "data": {**config["data"], **config["cam"]}},
            scene,
            gt_root,
            gt_masks_root,
            segment_every=config["semantic"].get("segment_every", 10),
            max_distance=config["semantic"].get("gt_mask_max_distance", 0.05),
            min_area=config["semantic"].get("gt_mask_min_area", 20),
            force=True,
        )

    if os.getenv('DISABLE_WANDB') == 'true':
        config["use_wandb"] = False
    elif config["use_wandb"]:
        wandb.init(
            project=config["project_name"],
            config=config,
            dir=str(path_utils.get_working_output_root() / "wandb"),
            group=config["data"]["scene_name"]
            if experiment_name != ""
            else experiment_name,
            name=f'{config["data"]["scene_name"]}_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}_{str(uuid.uuid4())[:5]}',
        )

    gen_utils.setup_seed(config["seed"])
    gslam = OVOSemMap(config, output_path=output_path)
    gslam.run()

    if tmp_run:
        final_path = path_utils.get_output_root() / dataset / experiment_name / scene
        shutil.move(output_path, final_path)

    if config["use_wandb"]:
        wandb.finish()
    print("Finished run.✨")

def main(args):
    if args.experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M")
        tmp_run = True
    else:
        assert len(args.experiment_name) > 0, "Experiment name cannot be '' "
        experiment_name = args.experiment_name
        tmp_run = False

    experiment_path = path_utils.get_output_root() / args.dataset_name / experiment_name

    if args.scenes_list is not None:
        with open(args.scenes_list, "r") as f:
            scenes = f.read().splitlines() 
    else:
        scenes = args.scenes

    if len(scenes) == 0 or args.segment or args.eval:
        path = path_utils.get_configs_root() / args.dataset_name / args.dataset_info_file
        with open(path, 'r') as f:
            dataset_info = yaml.full_load(f)

        if len(scenes) == 0:
            scenes = dataset_info["scenes"]

    for scene in scenes:        
        input_path = str(path_utils.get_datasets_root() / args.dataset_name / scene)
        if args.run:
            t0 = time.time()
            run_scene(
                scene,
                args.dataset_name,
                experiment_name,
                tmp_run=tmp_run,
                use_gt_masks=args.use_gt_masks,
                replica_gt_root=args.replica_gt_root,
                save_live_instance_vis=args.save_live_instance_vis,
                save_selective_prompt_debug=args.save_selective_prompt_debug,
                use_selective_prompt_points=args.use_selective_prompt_points,
            )
            t1 = time.time()
            print(f"Scene {scene} took: {t1-t0:.2f}")
        gc.collect()
 
    if args.segment: 
        data_path = str(path_utils.get_datasets_root())
        for scene in scenes:    
            scene_path = experiment_path / scene
            compute_scene_labels(scene_path, args.dataset_name, scene, data_path, dataset_info)

    if args.eval:
        if dataset_info["dataset"] == "scannet200":
            gt_path = Path(input_path).parent / "scannet200_gt"
        else:
            gt_path = Path(input_path).parent / "semantic_gt"
        eval_utils.eval_semantics(experiment_path / dataset_info["dataset"], gt_path, scenes, dataset_info, ignore_background=args.ignore_background)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Arguments to run and evaluate over a dataset')
    parser.add_argument('--dataset_name', help="Dataset used. Choose either `Replica`, `ScanNet`")
    parser.add_argument('--scenes', nargs="+", type=str, default=[], help=" List of scenes from given dataset to run.  If `--scenes_list` is set, this flag will be ignored.")
    parser.add_argument('--scenes_list',type=str, default=None, help="Path to a txt containing a scene name on each line. If set, `--scenes` is ignored. If neither `--scenes` nor `--scenes_list` are set, the scene list will be loaded from `<OVO_DATA_ROOT>/working/configs/<dataset_name>/<dataset_info_file>`")
    parser.add_argument('--dataset_info_file',type=str, default="eval_info.yaml")
    parser.add_argument('--experiment_name', default=None, type=str)
    parser.add_argument('--run', action='store_true', help="If set, compute the final metrics, after running OVO and segmenting.")
    parser.add_argument('--segment', action='store_true', help="If set, use the reconstructed scene to segment the gt point-cloud, after running OVO.")
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--ignore_background', action='store_true',help="If set, does not use background ids from eval_info to compute metrics.")
    parser.add_argument('--use_gt_masks', action='store_true', help="If set, build precomputed Replica GT instance masks and use them instead of SAM.")
    parser.add_argument('--replica_gt_root', type=str, default=None, help="Path to original Replica dataset root containing office_0/habitat/mesh_semantic.ply.")
    parser.add_argument('--save_live_instance_vis', action='store_true', help="If set, save per-frame debug images showing SAM and projected live 3D instance ids.")
    parser.add_argument('--save_selective_prompt_debug', action='store_true', help="If set, save known/unknown split debug images with object-wise known masks and sampled prompt points.")
    parser.add_argument('--use_selective_prompt_points', action='store_true', help="If set, replace SAM auto-grid points with the selective prompt points built from known/unknown regions.")
    args = parser.parse_args()
    main(args)
