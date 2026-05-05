from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from . import geometry_utils


def _to_numpy_image(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    return np.asarray(image, dtype=np.uint8)


def _to_numpy_seg_map(seg_map: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(seg_map, torch.Tensor):
        seg_map = seg_map.detach().cpu().numpy()
    return np.asarray(seg_map)


def _idx_to_rgb(
    idx_map: np.ndarray,
    *,
    max_idx: int = 40,
    zero_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    colours = plt.cm.tab20b.colors + plt.cm.tab20c.colors
    cmap = np.asarray(colours[:max_idx], dtype=np.float32)
    rgb = np.zeros((*idx_map.shape, 3), dtype=np.float32)
    valid_mask = idx_map >= 0
    if np.any(valid_mask):
        valid_ids = idx_map[valid_mask].astype(np.int64) % len(cmap)
        rgb[valid_mask] = cmap[valid_ids]
    if zero_color is not None:
        zero_mask = idx_map == 0
        if np.any(zero_mask):
            rgb[zero_mask] = np.asarray(zero_color, dtype=np.float32) / 255.0
    return (rgb * 255).astype(np.uint8)


def _blend_overlay(image: np.ndarray, overlay: np.ndarray, *, alpha: float = 0.6) -> np.ndarray:
    image_f = image.astype(np.float32)
    overlay_f = overlay.astype(np.float32)
    valid = np.any(overlay > 0, axis=-1, keepdims=True)
    blended = image_f.copy()
    blended[valid.squeeze(-1)] = (
        (1.0 - alpha) * image_f[valid.squeeze(-1)] + alpha * overlay_f[valid.squeeze(-1)]
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _render_projected_instance_overlay(
    image: np.ndarray,
    matches: np.ndarray,
    point_instance_ids: np.ndarray,
    *,
    point_radius: int = 1,
) -> np.ndarray:
    overlay = image.copy()
    if matches.size == 0:
        return overlay

    h, w = image.shape[:2]
    id_colors = _idx_to_rgb(
        point_instance_ids.reshape(-1, 1),
        zero_color=(255, 255, 255),
    ).reshape(-1, 3)
    for (x, y), color, instance_id in zip(matches, id_colors, point_instance_ids):
        if instance_id < 0:
            continue
        x0 = max(int(x) - point_radius, 0)
        x1 = min(int(x) + point_radius + 1, w)
        y0 = max(int(y) - point_radius, 0)
        y1 = min(int(y) + point_radius + 1, h)
        overlay[y0:y1, x0:x1] = color
    return overlay


def build_live_instance_debug_image(
    *,
    image: np.ndarray | torch.Tensor,
    depth: np.ndarray | torch.Tensor,
    points_3d: torch.Tensor,
    point_instance_ids: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    seg_map: np.ndarray | torch.Tensor,
    match_distance_th: float,
    rgb_depth_ratio: tuple[float, float, int] = (),
) -> np.ndarray:
    rgb_image = _to_numpy_image(image)
    seg_map_np = _to_numpy_seg_map(seg_map)
    depth_t = depth if isinstance(depth, torch.Tensor) else torch.from_numpy(np.asarray(depth))
    depth_t = depth_t.to(device=points_3d.device, dtype=torch.float32)
    w2c = torch.linalg.inv(c2w)
    matched_point_idxs, matches = geometry_utils.match_3d_points_to_2d_pixels(
        depth_t,
        w2c,
        points_3d,
        intrinsics,
        match_distance_th,
    )

    if len(rgb_depth_ratio) > 0 and matches.numel() > 0:
        matches = matches.clone()
        matches += rgb_depth_ratio[-1]
        matches[:, 1] = (matches[:, 1] * rgb_depth_ratio[0]).int()
        matches[:, 0] = (matches[:, 0] * rgb_depth_ratio[1]).int()

    point_instance_ids = point_instance_ids.reshape(-1)
    matched_instance_ids = point_instance_ids[matched_point_idxs].detach().cpu().numpy()
    matches_np = matches.detach().cpu().numpy() if matches.numel() > 0 else np.empty((0, 2), dtype=np.int32)

    sam_rgb = _idx_to_rgb(seg_map_np)
    sam_overlay = _blend_overlay(rgb_image, sam_rgb)
    live_overlay = _render_projected_instance_overlay(rgb_image, matches_np, matched_instance_ids)

    top_row = np.concatenate((rgb_image, sam_rgb), axis=1)
    bottom_row = np.concatenate((sam_overlay, live_overlay), axis=1)
    return np.concatenate((top_row, bottom_row), axis=0)


def build_projected_instance_overlay_image(
    *,
    image: np.ndarray | torch.Tensor,
    depth: np.ndarray | torch.Tensor,
    points_3d: torch.Tensor,
    point_instance_ids: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    match_distance_th: float,
    rgb_depth_ratio: tuple[float, float, int] = (),
) -> np.ndarray:
    rgb_image = _to_numpy_image(image)
    depth_t = depth if isinstance(depth, torch.Tensor) else torch.from_numpy(np.asarray(depth))
    depth_t = depth_t.to(device=points_3d.device, dtype=torch.float32)
    w2c = torch.linalg.inv(c2w)
    matched_point_idxs, matches = geometry_utils.match_3d_points_to_2d_pixels(
        depth_t,
        w2c,
        points_3d,
        intrinsics,
        match_distance_th,
    )

    if len(rgb_depth_ratio) > 0 and matches.numel() > 0:
        matches = matches.clone()
        matches += rgb_depth_ratio[-1]
        matches[:, 1] = (matches[:, 1] * rgb_depth_ratio[0]).int()
        matches[:, 0] = (matches[:, 0] * rgb_depth_ratio[1]).int()

    point_instance_ids = point_instance_ids.reshape(-1)
    matched_instance_ids = point_instance_ids[matched_point_idxs].detach().cpu().numpy()
    matches_np = matches.detach().cpu().numpy() if matches.numel() > 0 else np.empty((0, 2), dtype=np.int32)
    return _render_projected_instance_overlay(rgb_image, matches_np, matched_instance_ids)


def save_live_instance_debug_image(
    *,
    output_dir: str | Path,
    frame_id: int,
    image: np.ndarray | torch.Tensor,
    depth: np.ndarray | torch.Tensor,
    points_3d: torch.Tensor,
    point_instance_ids: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    seg_map: np.ndarray | torch.Tensor,
    match_distance_th: float,
    rgb_depth_ratio: tuple[float, float, int] = (),
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_image = build_live_instance_debug_image(
        image=image,
        depth=depth,
        points_3d=points_3d,
        point_instance_ids=point_instance_ids,
        intrinsics=intrinsics,
        c2w=c2w,
        seg_map=seg_map,
        match_distance_th=match_distance_th,
        rgb_depth_ratio=rgb_depth_ratio,
    )
    output_path = output_dir / f"{frame_id:04d}_live_instance_debug.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
    return output_path


def save_projected_instance_overlay_image(
    *,
    output_dir: str | Path,
    frame_id: int,
    image: np.ndarray | torch.Tensor,
    depth: np.ndarray | torch.Tensor,
    points_3d: torch.Tensor,
    point_instance_ids: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    match_distance_th: float,
    rgb_depth_ratio: tuple[float, float, int] = (),
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_image = build_projected_instance_overlay_image(
        image=image,
        depth=depth,
        points_3d=points_3d,
        point_instance_ids=point_instance_ids,
        intrinsics=intrinsics,
        c2w=c2w,
        match_distance_th=match_distance_th,
        rgb_depth_ratio=rgb_depth_ratio,
    )
    output_path = output_dir / f"{frame_id:04d}_projected_object_ids.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
    return output_path
