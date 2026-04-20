from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os

import numpy as np


def build_vertex_object_ids(
    num_vertices: int,
    faces: np.ndarray,
    face_object_ids: np.ndarray,
    invalid_id: int = -1,
) -> np.ndarray:
    vertex_votes: list[dict[int, int]] = [dict() for _ in range(num_vertices)]
    for face, object_id in zip(faces, face_object_ids):
        object_id = int(object_id)
        if object_id < 0:
            continue
        for vertex_idx in face:
            votes = vertex_votes[int(vertex_idx)]
            votes[object_id] = votes.get(object_id, 0) + 1

    vertex_object_ids = np.full(num_vertices, invalid_id, dtype=np.int32)
    for vertex_idx, votes in enumerate(vertex_votes):
        if not votes:
            continue
        vertex_object_ids[vertex_idx] = max(votes.items(), key=lambda item: (item[1], -item[0]))[0]
    return vertex_object_ids


def object_id_map_to_masks(
    object_id_map: np.ndarray,
    min_area: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seg_map = -np.ones(object_id_map.shape, dtype=np.int32)
    binary_maps = []
    object_ids = []

    valid_object_ids = object_id_map[object_id_map >= 0]
    if valid_object_ids.size == 0:
        return seg_map, np.zeros((0, *object_id_map.shape), dtype=bool), np.zeros((0,), dtype=np.int32)

    unique_ids, counts = np.unique(valid_object_ids, return_counts=True)
    ordered_ids = [int(obj_id) for obj_id, count in sorted(zip(unique_ids, counts), key=lambda item: (-item[1], item[0]))]
    for object_id in ordered_ids:
        mask = object_id_map == object_id
        if int(mask.sum()) < min_area:
            continue
        seg_idx = len(binary_maps)
        seg_map[mask] = seg_idx
        binary_maps.append(mask)
        object_ids.append(object_id)

    if not binary_maps:
        return seg_map, np.zeros((0, *object_id_map.shape), dtype=bool), np.zeros((0,), dtype=np.int32)

    return seg_map, np.stack(binary_maps, axis=0), np.asarray(object_ids, dtype=np.int32)


def backproject_depth_to_world(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    if valid_mask is None:
        valid_mask = depth > 0
    if not np.any(valid_mask):
        return np.zeros((0, 3), dtype=np.float32)

    ys, xs = np.nonzero(valid_mask)
    zs = depth[ys, xs].astype(np.float32)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    xs_cam = (xs.astype(np.float32) - cx) * zs / fx
    ys_cam = (ys.astype(np.float32) - cy) * zs / fy
    points_cam = np.stack([xs_cam, ys_cam, zs], axis=1)
    rotation = c2w[:3, :3].astype(np.float32)
    translation = c2w[:3, 3].astype(np.float32)
    return points_cam @ rotation.T + translation


def scene_to_replica_scene(scene_name: str) -> str:
    if scene_name.startswith("office"):
        suffix = scene_name[len("office") :]
        return f"office_{suffix}"
    if scene_name.startswith("room"):
        suffix = scene_name[len("room") :]
        return f"room_{suffix}"
    return scene_name


def load_replica_semantic_vertices(replica_scene_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    from plyfile import PlyData

    ply = PlyData.read(str(replica_scene_dir / "habitat" / "mesh_semantic.ply"))
    vertices = np.stack(
        [
            np.asarray(ply["vertex"]["x"], dtype=np.float32),
            np.asarray(ply["vertex"]["y"], dtype=np.float32),
            np.asarray(ply["vertex"]["z"], dtype=np.float32),
        ],
        axis=1,
    )
    faces = np.vstack(ply["face"]["vertex_indices"]).astype(np.int64)
    face_object_ids = np.asarray(ply["face"]["object_id"], dtype=np.int32)
    vertex_object_ids = build_vertex_object_ids(len(vertices), faces, face_object_ids)
    return vertices, vertex_object_ids


def _object_id_map_from_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    vertex_tree: Any,
    vertex_object_ids: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    valid_mask = depth > 0
    object_id_map = -np.ones(depth.shape, dtype=np.int32)
    if not np.any(valid_mask):
        return object_id_map

    world_points = backproject_depth_to_world(depth, intrinsics, c2w, valid_mask=valid_mask)
    distances, vertex_indices = vertex_tree.query(world_points, workers=-1)
    distances = np.asarray(distances, dtype=np.float32)
    vertex_indices = np.asarray(vertex_indices, dtype=np.int64)

    ys, xs = np.nonzero(valid_mask)
    matched = distances <= max_distance
    if np.any(matched):
        object_id_map[ys[matched], xs[matched]] = vertex_object_ids[vertex_indices[matched]]
    return object_id_map


def _seg_idx_to_rgb(seg_map: np.ndarray, max_idx: int = 40) -> np.ndarray:
    import matplotlib.pyplot as plt

    colours = plt.cm.tab20b.colors + plt.cm.tab20c.colors
    cmap = np.asarray(colours[:max_idx], dtype=np.float32)
    rgb = np.zeros((*seg_map.shape, 3), dtype=np.float32)
    if seg_map.size == 0 or seg_map.max() < 0:
        return rgb.astype(np.uint8)
    for idx in range(int(seg_map.max()) + 1):
        mask = seg_map == idx
        if np.any(mask):
            rgb[mask] = cmap[idx % len(cmap)]
    return (rgb * 255).astype(np.uint8)


def _blend_overlay(image: np.ndarray, seg_rgb: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    image_f = image.astype(np.float32)
    seg_f = seg_rgb.astype(np.float32)
    valid = np.any(seg_rgb > 0, axis=-1)
    blended = image_f.copy()
    blended[valid] = (1.0 - alpha) * image_f[valid] + alpha * seg_f[valid]
    return np.clip(blended, 0, 255).astype(np.uint8)


def _save_gt_masks(
    output_dir: Path,
    frame_id: int,
    seg_map: np.ndarray,
    binary_maps: np.ndarray,
    object_ids: np.ndarray,
    image: np.ndarray | None = None,
) -> None:
    import imageio.v2 as imageio

    np.save(output_dir / f"{frame_id:04d}_seg_map_default.npy", seg_map)
    np.save(output_dir / f"{frame_id:04d}_bmap_default.npy", binary_maps)
    np.save(output_dir / f"{frame_id:04d}_object_ids.npy", object_ids)
    if image is not None:
        imageio.imwrite(output_dir / f"{frame_id:04d}_rgb.png", image.astype(np.uint8))
        seg_rgb = _seg_idx_to_rgb(seg_map)
        imageio.imwrite(output_dir / f"{frame_id:04d}_seg_map_default.png", seg_rgb)
        imageio.imwrite(output_dir / f"{frame_id:04d}_seg_overlay_default.png", _blend_overlay(image, seg_rgb))


def precompute_replica_gt_masks(
    dataset_config: dict[str, Any],
    scene_name: str,
    replica_root: str | Path,
    output_dir: str | Path,
    *,
    segment_every: int = 10,
    max_distance: float = 0.05,
    min_area: int = 20,
    force: bool = False,
) -> Path:
    import tqdm
    from scipy.spatial import cKDTree

    from ..entities.datasets import get_dataset

    replica_scene_dir = Path(replica_root) / scene_to_replica_scene(scene_name)
    if not replica_scene_dir.exists():
        raise FileNotFoundError(f"Replica GT scene not found: {replica_scene_dir}")

    output_dir = Path(output_dir) / scene_name
    os.makedirs(output_dir, exist_ok=True)
    dataset = get_dataset(dataset_config["dataset_name"])(dataset_config["data"])
    vertices, vertex_object_ids = load_replica_semantic_vertices(replica_scene_dir)
    vertex_tree = cKDTree(vertices)

    frame_ids = [frame_id for frame_id in range(len(dataset)) if frame_id % segment_every == 0]
    for frame_id in tqdm.tqdm(frame_ids, desc=f"GT masks {scene_name}"):
        seg_map_path = output_dir / f"{frame_id:04d}_seg_map_default.npy"
        bmap_path = output_dir / f"{frame_id:04d}_bmap_default.npy"
        if not force and seg_map_path.exists() and bmap_path.exists():
            continue
        _, image, depth, c2w = dataset[frame_id]
        object_id_map = _object_id_map_from_depth(
            depth,
            dataset.intrinsics,
            c2w,
            vertex_tree,
            vertex_object_ids,
            max_distance=max_distance,
        )
        seg_map, binary_maps, object_ids = object_id_map_to_masks(object_id_map, min_area=min_area)
        _save_gt_masks(output_dir, frame_id, seg_map, binary_maps, object_ids, image=image)

    metadata = {
        "scene_name": scene_name,
        "replica_scene_dir": str(replica_scene_dir),
        "segment_every": int(segment_every),
        "max_distance": float(max_distance),
        "min_area": int(min_area),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return output_dir
