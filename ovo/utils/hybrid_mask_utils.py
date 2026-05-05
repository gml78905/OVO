from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import time

import cv2
import numpy as np
import torch

from . import geometry_utils


def _perf_now() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _project_points_with_instance_ids(
    points_3d: torch.Tensor,
    points_ins_ids: torch.Tensor,
    depth: np.ndarray | torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    rgb_depth_ratio: Tuple[float, float, int] = (),
    match_distance_th: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if points_3d.numel() == 0 or points_ins_ids.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=points_ins_ids.device),
            torch.empty((0, 2), dtype=torch.int64, device=points_ins_ids.device),
        )

    depth_t = depth if isinstance(depth, torch.Tensor) else torch.from_numpy(np.asarray(depth))
    depth_t = depth_t.to(device=points_3d.device, dtype=torch.float32)
    matched_point_idxs, matches = geometry_utils.match_3d_points_to_2d_pixels(
        depth_t,
        torch.linalg.inv(c2w),
        points_3d,
        intrinsics,
        match_distance_th,
    )
    if matches.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=points_ins_ids.device),
            torch.empty((0, 2), dtype=torch.int64, device=points_ins_ids.device),
        )

    if len(rgb_depth_ratio) > 0:
        matches = matches.clone()
        matches += rgb_depth_ratio[-1]
        matches[:, 1] = (matches[:, 1] * rgb_depth_ratio[0]).int()
        matches[:, 0] = (matches[:, 0] * rgb_depth_ratio[1]).int()

    matched_ins_ids = points_ins_ids.reshape(-1)[matched_point_idxs]
    return matched_ins_ids, matches


def _build_mask_from_matches(
    matches: torch.Tensor,
    image_shape: Tuple[int, int],
    dilation_kernel: int,
    closing_kernel: int,
) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    coords = matches.detach().cpu().numpy().astype(np.int32)
    coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
    mask[coords[:, 1], coords[:, 0]] = 1
    return _refine_binary_mask(mask.astype(bool), dilation_kernel, closing_kernel)


def build_known_object_masks(
    points_3d: torch.Tensor,
    points_ins_ids: torch.Tensor,
    depth: np.ndarray | torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    image_shape: Tuple[int, int],
    rgb_depth_ratio: Tuple[float, float, int] = (),
    match_distance_th: float = 0.05,
    min_points: int = 1,
    dilation_kernel: int = 5,
    closing_kernel: int = 5,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    height, width = image_shape
    known_masks: Dict[int, np.ndarray] = {}
    known_union = np.zeros((height, width), dtype=bool)

    if points_3d.numel() == 0 or points_ins_ids.numel() == 0:
        return known_masks, known_union

    matched_ins_ids, matches = _project_points_with_instance_ids(
        points_3d,
        points_ins_ids,
        depth,
        intrinsics,
        c2w,
        rgb_depth_ratio=rgb_depth_ratio,
        match_distance_th=match_distance_th,
    )
    if matches.numel() == 0:
        return known_masks, known_union

    valid = matched_ins_ids > 0
    if not torch.any(valid):
        return known_masks, known_union

    matches = matches[valid]
    matched_ins_ids = matched_ins_ids[valid]

    for ins_id in torch.unique(matched_ins_ids, sorted=True).tolist():
        ins_matches = matches[matched_ins_ids == ins_id]
        if ins_matches.shape[0] < min_points:
            continue
        refined = _build_mask_from_matches(ins_matches, image_shape, dilation_kernel, closing_kernel)
        known_masks[int(ins_id)] = refined
        known_union = np.logical_or(known_union, refined)

    return known_masks, known_union


def build_seen_unknown_mask(
    points_3d: torch.Tensor,
    points_ins_ids: torch.Tensor,
    depth: np.ndarray | torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    image_shape: Tuple[int, int],
    rgb_depth_ratio: Tuple[float, float, int] = (),
    match_distance_th: float = 0.05,
    min_points: int = 1,
    dilation_kernel: int = 5,
    closing_kernel: int = 5,
) -> np.ndarray:
    matched_ins_ids, matches = _project_points_with_instance_ids(
        points_3d,
        points_ins_ids,
        depth,
        intrinsics,
        c2w,
        rgb_depth_ratio=rgb_depth_ratio,
        match_distance_th=match_distance_th,
    )
    if matches.numel() == 0:
        return np.zeros(image_shape, dtype=bool)
    seen_unknown_matches = matches[matched_ins_ids == 0]
    if seen_unknown_matches.shape[0] < min_points:
        return np.zeros(image_shape, dtype=bool)
    return _build_mask_from_matches(seen_unknown_matches, image_shape, dilation_kernel, closing_kernel)


def sample_seen_unknown_component_points(
    mask: np.ndarray,
    *,
    cell_size_px: float,
    min_area: int,
) -> np.ndarray:
    component_infos = analyze_seen_unknown_components(
        mask,
        cell_size_px=cell_size_px,
        min_area=min_area,
    )
    kept_points = [np.asarray(info["points"], dtype=np.float32) for info in component_infos if not info["skipped"]]
    if not kept_points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(kept_points, axis=0)


def analyze_seen_unknown_components(
    mask: np.ndarray,
    *,
    cell_size_px: float,
    min_area: int,
) -> List[Dict[str, object]]:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return []

    labels_n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
    cell_area = max(float(cell_size_px) * float(cell_size_px), 1.0)
    component_infos: List[Dict[str, object]] = []

    for label_id in range(1, labels_n):
        component_mask = labels == label_id
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        info: Dict[str, object] = {
            "component_id": int(label_id),
            "mask": component_mask,
            "area": area,
            "skipped": area < min_area,
            "skip_reason": "small_area" if area < min_area else "",
            "n_points": 0,
            "margin_px": 0,
            "max_dist": 0.0,
            "points": np.zeros((0, 2), dtype=np.float32),
        }
        if area < min_area:
            component_infos.append(info)
            continue
        dist = _compute_interior_distance(np.asarray(component_mask, dtype=bool))
        max_dist = float(dist.max()) if dist.size > 0 else 0.0
        n_points, margin_px = _seen_unknown_sampling_policy(
            component_mask=component_mask,
            area=area,
            cell_area=cell_area,
            cell_size_px=cell_size_px,
        )
        sampled = _sample_inward_contour_points(component_mask, n_points=n_points, margin_px=margin_px)
        info["n_points"] = int(n_points)
        info["margin_px"] = int(margin_px)
        info["max_dist"] = max_dist
        info["points"] = sampled.astype(np.float32) if sampled.size > 0 else np.zeros((0, 2), dtype=np.float32)
        component_infos.append(info)

    return component_infos


def compute_unknown_components(
    depth: np.ndarray | torch.Tensor,
    known_union_mask: np.ndarray,
    min_area: int = 100,
    min_peak_distance: float = 1.0,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    depth_np = depth.detach().cpu().numpy() if isinstance(depth, torch.Tensor) else np.asarray(depth)
    valid_depth_mask = depth_np > 0
    unknown_mask = np.logical_and(valid_depth_mask, np.logical_not(np.asarray(known_union_mask, dtype=bool)))
    labels_n, labels, stats, _ = cv2.connectedComponentsWithStats(unknown_mask.astype(np.uint8), connectivity=8)

    regions: List[Dict[str, object]] = []
    filtered_unknown = np.zeros_like(unknown_mask, dtype=bool)
    for label_id in range(1, labels_n):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        region_mask = labels == label_id
        filtered_unknown[region_mask] = True
        regions.append(
            {
                "region_id": int(label_id),
                "area": area,
                "mask": region_mask,
            }
        )
    return filtered_unknown, regions


def sample_mask_component_points(mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return np.zeros((0, 2), dtype=np.float32)

    labels_n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
    component_areas = [int(stats[label_id, cv2.CC_STAT_AREA]) for label_id in range(1, labels_n)]
    area_threshold = 0
    if len(component_areas) > 1:
        area_threshold = int(np.ceil(max(component_areas) * 0.25))
    points: List[List[float]] = []
    for label_id in range(1, labels_n):
        if area_threshold > 0 and int(stats[label_id, cv2.CC_STAT_AREA]) < area_threshold:
            continue
        point = _sample_single_mask_point(labels == label_id)
        if point is not None:
            points.append(point.tolist())
    return np.asarray(points, dtype=np.float32) if points else np.zeros((0, 2), dtype=np.float32)


def sample_unknown_region_points(
    region_mask: np.ndarray,
    cell_size_px: float,
    area_per_point: int = 4000,
    max_points: int = 8,
    min_peak_distance: float = 1.0,
) -> np.ndarray:
    mask_bool = np.asarray(region_mask, dtype=bool)
    if not np.any(mask_bool):
        return np.zeros((0, 2), dtype=np.float32)
    ys, xs = np.nonzero(mask_bool)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    cell_size = max(float(cell_size_px), 1.0)
    n_cols = max(1, int(np.ceil((x_max - x_min + 1) / cell_size)))
    n_rows = max(1, int(np.ceil((y_max - y_min + 1) / cell_size)))
    image_h, image_w = mask_bool.shape

    selected: List[List[float]] = []
    for row_idx in range(n_rows):
        y_start = y_min + row_idx * cell_size
        center_y = int(round(y_start + 0.5 * cell_size))
        if center_y < 0 or center_y >= image_h:
            continue
        for col_idx in range(n_cols):
            x_start = x_min + col_idx * cell_size
            center_x = int(round(x_start + 0.5 * cell_size))
            if center_x < 0 or center_x >= image_w:
                continue
            if mask_bool[center_y, center_x]:
                selected.append([float(center_x), float(center_y)])

    return np.asarray(selected, dtype=np.float32) if selected else np.zeros((0, 2), dtype=np.float32)


def build_selective_prompt_plan(
    points_3d: torch.Tensor,
    points_ins_ids: torch.Tensor,
    depth: np.ndarray | torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    image_shape: Tuple[int, int],
    rgb_depth_ratio: Tuple[float, float, int] = (),
    match_distance_th: float = 0.05,
    known_min_points: int = 1,
    known_dilation_kernel: int = 5,
    known_closing_kernel: int = 5,
    unknown_min_area: int = 100,
    unknown_area_per_point: int = 4000,
    unknown_max_points: int = 8,
    unknown_min_peak_distance: float = 1.0,
    unknown_grid_cells_per_width: int = 16,
) -> Dict[str, object]:
    prompt_plan, _ = build_selective_prompt_plan_timed(
        points_3d=points_3d,
        points_ins_ids=points_ins_ids,
        depth=depth,
        intrinsics=intrinsics,
        c2w=c2w,
        image_shape=image_shape,
        rgb_depth_ratio=rgb_depth_ratio,
        match_distance_th=match_distance_th,
        known_min_points=known_min_points,
        known_dilation_kernel=known_dilation_kernel,
        known_closing_kernel=known_closing_kernel,
        unknown_min_area=unknown_min_area,
        unknown_area_per_point=unknown_area_per_point,
        unknown_max_points=unknown_max_points,
        unknown_min_peak_distance=unknown_min_peak_distance,
        unknown_grid_cells_per_width=unknown_grid_cells_per_width,
    )
    return prompt_plan


def build_selective_prompt_plan_timed(
    points_3d: torch.Tensor,
    points_ins_ids: torch.Tensor,
    depth: np.ndarray | torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    image_shape: Tuple[int, int],
    rgb_depth_ratio: Tuple[float, float, int] = (),
    match_distance_th: float = 0.05,
    known_min_points: int = 1,
    known_dilation_kernel: int = 5,
    known_closing_kernel: int = 5,
    unknown_min_area: int = 100,
    unknown_area_per_point: int = 4000,
    unknown_max_points: int = 8,
    unknown_min_peak_distance: float = 1.0,
    unknown_grid_cells_per_width: int = 16,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    unknown_cell_size_px = image_shape[1] / float(max(unknown_grid_cells_per_width, 1))
    unknown_min_region_area = max(1, int(np.ceil((unknown_cell_size_px * unknown_cell_size_px) / 4.0)))

    t0 = _perf_now()
    known_masks, known_union = build_known_object_masks(
        points_3d,
        points_ins_ids,
        depth,
        intrinsics,
        c2w,
        image_shape,
        rgb_depth_ratio=rgb_depth_ratio,
        match_distance_th=match_distance_th,
        min_points=known_min_points,
        dilation_kernel=known_dilation_kernel,
        closing_kernel=known_closing_kernel,
    )
    t1 = _perf_now()
    known_points = {
        ins_id: sample_mask_component_points(mask)
        for ins_id, mask in known_masks.items()
    }
    t2 = _perf_now()
    seen_unknown_mask = build_seen_unknown_mask(
        points_3d,
        points_ins_ids,
        depth,
        intrinsics,
        c2w,
        image_shape,
        rgb_depth_ratio=rgb_depth_ratio,
        match_distance_th=match_distance_th,
        min_points=known_min_points,
        dilation_kernel=known_dilation_kernel,
        closing_kernel=known_closing_kernel,
    )
    t2a = _perf_now()
    seen_unknown_components = analyze_seen_unknown_components(
        seen_unknown_mask,
        cell_size_px=unknown_cell_size_px,
        min_area=unknown_min_region_area,
    )
    t2b = _perf_now()
    seen_unknown_points = sample_seen_unknown_component_points(
        seen_unknown_mask,
        cell_size_px=unknown_cell_size_px,
        min_area=unknown_min_region_area,
    )
    t3 = _perf_now()

    unknown_mask, unknown_regions = compute_unknown_components(
        depth,
        np.logical_or(known_union, seen_unknown_mask),
        min_area=unknown_min_region_area,
        min_peak_distance=unknown_min_peak_distance,
    )
    for region in unknown_regions:
        region["points"] = sample_unknown_region_points(
            np.asarray(region["mask"], dtype=bool),
            cell_size_px=unknown_cell_size_px,
            area_per_point=unknown_area_per_point,
            max_points=unknown_max_points,
            min_peak_distance=unknown_min_peak_distance,
        )
    t4 = _perf_now()

    prompt_plan = {
        "known_masks": known_masks,
        "known_union_mask": known_union,
        "known_points": known_points,
        "seen_unknown_mask": seen_unknown_mask,
        "seen_unknown_points": seen_unknown_points,
        "seen_unknown_components": seen_unknown_components,
        "brand_new_unknown_mask": unknown_mask,
        "unknown_mask": unknown_mask,
        "unknown_regions": unknown_regions,
        "unknown_cell_size_px": unknown_cell_size_px,
        "unknown_min_region_area": unknown_min_region_area,
    }
    _ = collect_all_prompt_points(prompt_plan)
    t5 = _perf_now()

    timings = {
        "t_prompt_proj_known": round(t1 - t0, 6),
        "t_prompt_known": round(t2 - t1, 6),
        "t_prompt_seen_unknown": round(t3 - t2, 6),
        "t_prompt_seen_unknown_mask": round(t2a - t2, 6),
        "t_prompt_seen_unknown_components": round(t2b - t2a, 6),
        "t_prompt_seen_unknown_sample": round(t3 - t2b, 6),
        "t_prompt_brand_new_unknown": round(t4 - t3, 6),
        "t_prompt_collect": round(t5 - t4, 6),
        "t_prompt_total": round(t5 - t0, 6),
    }
    return prompt_plan, timings


def collect_all_prompt_points(prompt_plan: Dict[str, object]) -> np.ndarray:
    all_points: List[np.ndarray] = []

    known_points = prompt_plan.get("known_points", {})
    if isinstance(known_points, dict):
        for points in known_points.values():
            pts = np.asarray(points, dtype=np.float32)
            if pts.size > 0:
                all_points.append(pts.reshape(-1, 2))

    seen_unknown_points = np.asarray(
        prompt_plan.get("seen_unknown_points", np.zeros((0, 2), dtype=np.float32)),
        dtype=np.float32,
    )
    if seen_unknown_points.size > 0:
        all_points.append(seen_unknown_points.reshape(-1, 2))

    for region in prompt_plan.get("unknown_regions", []):
        pts = np.asarray(region.get("points", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        if pts.size > 0:
            all_points.append(pts.reshape(-1, 2))

    if not all_points:
        return np.zeros((0, 2), dtype=np.float32)

    stacked = np.concatenate(all_points, axis=0)
    rounded = np.round(stacked).astype(np.int32)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return stacked[unique_indices].astype(np.float32)


def save_selective_prompt_debug_views(
    output_dir: str | Path,
    frame_id: int,
    image: np.ndarray,
    prompt_plan: Dict[str, object],
    sam_surviving_points: np.ndarray | None = None,
    sam_seg_map: np.ndarray | None = None,
    projected_object_ids_view: np.ndarray | None = None,
) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    known_masks = prompt_plan["known_masks"]
    known_union_mask = np.asarray(prompt_plan["known_union_mask"], dtype=bool)
    known_points = prompt_plan["known_points"]
    seen_unknown_mask = np.asarray(
        prompt_plan.get("seen_unknown_mask", np.zeros_like(known_union_mask, dtype=bool)),
        dtype=bool,
    )
    seen_unknown_points = np.asarray(
        prompt_plan.get("seen_unknown_points", np.zeros((0, 2), dtype=np.float32)),
        dtype=np.float32,
    )
    brand_new_unknown_mask = np.asarray(
        prompt_plan.get("brand_new_unknown_mask", prompt_plan["unknown_mask"]),
        dtype=bool,
    )
    unknown_mask = np.asarray(prompt_plan["unknown_mask"], dtype=bool)
    unknown_regions = prompt_plan["unknown_regions"]

    saved_paths: List[Path] = []
    summary_views = _render_selective_prompt_views(
        image=image,
        known_masks=known_masks,
        known_union_mask=known_union_mask,
        known_points=known_points,
        seen_unknown_mask=seen_unknown_mask,
        seen_unknown_points=seen_unknown_points,
        brand_new_unknown_mask=brand_new_unknown_mask,
        unknown_mask=unknown_mask,
        unknown_regions=unknown_regions,
        sam_surviving_points=sam_surviving_points,
        sam_seg_map=sam_seg_map,
        projected_object_ids_view=projected_object_ids_view,
    )

    summary_image = summary_views["summary"]
    path = output_dir / f"{frame_id:04d}_summary.png"
    cv2.imwrite(str(path), cv2.cvtColor(summary_image, cv2.COLOR_RGB2BGR))
    saved_paths.append(path)

    return saved_paths


def build_projected_bbox_prompts(
    points_3d: torch.Tensor,
    points_ins_ids: torch.Tensor,
    intrinsics: torch.Tensor,
    w2c: torch.Tensor,
    image_shape: Tuple[int, int],
    rgb_depth_ratio: Tuple[float, float, int] = (),
    min_points: int = 1,
    padding: int = 0,
) -> List[Dict[str, np.ndarray | int]]:
    if points_3d.numel() == 0:
        return []

    if points_3d.shape[-1] == 3:
        ones = torch.ones((points_3d.shape[0], 1), dtype=points_3d.dtype, device=points_3d.device)
        points_3d = torch.cat([points_3d, ones], dim=1)

    local_points = torch.einsum("mn,bn->bm", w2c, points_3d)
    in_front_mask = local_points[:, 2] > 0
    if not torch.any(in_front_mask):
        return []

    local_points = local_points[in_front_mask]
    local_ins_ids = points_ins_ids[in_front_mask]
    points_2d = geometry_utils.project_3d_points(local_points, intrinsics).to(torch.float32)

    if len(rgb_depth_ratio) > 0:
        crop_edge = rgb_depth_ratio[-1]
        points_2d += crop_edge
        points_2d[:, 1] *= rgb_depth_ratio[0]
        points_2d[:, 0] *= rgb_depth_ratio[1]

    height, width = image_shape
    in_plane_mask = (
        (points_2d[:, 0] >= 0)
        & (points_2d[:, 0] < width)
        & (points_2d[:, 1] >= 0)
        & (points_2d[:, 1] < height)
        & (local_ins_ids > 0)
    )
    if not torch.any(in_plane_mask):
        return []

    points_2d = points_2d[in_plane_mask]
    local_ins_ids = local_ins_ids[in_plane_mask]

    prompts: List[Dict[str, np.ndarray | int]] = []
    for ins_id in torch.unique(local_ins_ids, sorted=True).tolist():
        ins_points = points_2d[local_ins_ids == ins_id]
        if ins_points.shape[0] < min_points:
            continue
        x1 = max(0.0, ins_points[:, 0].min().item() - padding)
        y1 = max(0.0, ins_points[:, 1].min().item() - padding)
        x2 = min(float(width - 1), ins_points[:, 0].max().item() + padding)
        y2 = min(float(height - 1), ins_points[:, 1].max().item() + padding)
        prompts.append(
            {
                "ins_id": int(ins_id),
                "box": np.array([x1, y1, x2, y2], dtype=np.float32),
            }
        )
    return prompts


def filter_point_masks_by_coverage(
    point_masks: Sequence[Dict[str, object]],
    covered_mask: np.ndarray,
    min_component_area: int = 1,
) -> Tuple[List[Dict[str, object]], np.ndarray]:
    filtered_masks: List[Dict[str, object]] = []
    updated_coverage = covered_mask.copy()
    valid_uncovered = valid_uncovered_mask(covered_mask, min_component_area)

    for mask in point_masks:
        point = _extract_point(mask)
        if point is None:
            filtered_masks.append(mask)
            updated_coverage = np.logical_or(updated_coverage, np.asarray(mask["segmentation"], dtype=bool))
            continue

        x, y = point
        if y < 0 or x < 0 or y >= valid_uncovered.shape[0] or x >= valid_uncovered.shape[1]:
            continue
        if not valid_uncovered[y, x]:
            continue
        if updated_coverage[y, x]:
            continue
        filtered_masks.append(mask)
        updated_coverage = np.logical_or(updated_coverage, np.asarray(mask["segmentation"], dtype=bool))

    return filtered_masks, updated_coverage


def valid_uncovered_mask(covered_mask: np.ndarray, min_component_area: int) -> np.ndarray:
    uncovered = np.logical_not(np.asarray(covered_mask, dtype=bool))
    if min_component_area <= 1:
        return uncovered
    if not np.any(uncovered):
        return uncovered

    labels_n, labels, stats, _ = cv2.connectedComponentsWithStats(uncovered.astype(np.uint8), connectivity=8)
    if labels_n <= 1:
        return np.zeros_like(uncovered, dtype=bool)

    valid = np.zeros_like(uncovered, dtype=bool)
    for label_id in range(1, labels_n):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            valid[labels == label_id] = True
    return valid


def filter_point_grid_for_sam(
    point_grid_norm: np.ndarray,
    image_shape: Tuple[int, int],
    covered_mask: np.ndarray,
    min_component_area: int = 1,
    min_distance_to_covered: float = 0.0,
) -> np.ndarray:
    points = np.asarray(point_grid_norm, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    h, w = image_shape
    valid_region = valid_uncovered_mask(covered_mask, min_component_area)
    if min_distance_to_covered > 0 and np.any(covered_mask):
        uncovered = np.logical_not(np.asarray(covered_mask, dtype=bool)).astype(np.uint8)
        dist_to_covered = cv2.distanceTransform(uncovered, distanceType=cv2.DIST_L2, maskSize=3)
        valid_region = np.logical_and(valid_region, dist_to_covered > float(min_distance_to_covered))

    px = np.clip(np.round(points[:, 0] * float(w)).astype(np.int32), 0, w - 1)
    py = np.clip(np.round(points[:, 1] * float(h)).astype(np.int32), 0, h - 1)
    keep = valid_region[py, px]
    return points[keep]


def merge_bbox_point_masks(
    bbox_masks: Sequence[Dict[str, object]],
    point_masks: Sequence[Dict[str, object]],
    overlap_th: float = 0.7,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    merged_bbox_masks = [dict(mask) for mask in bbox_masks]
    remaining_point_masks: List[Dict[str, object]] = []

    for point_mask in point_masks:
        point_seg = np.asarray(point_mask["segmentation"], dtype=bool)
        point_area = int(point_seg.sum())
        if point_area == 0:
            continue

        best_bbox_idx = -1
        best_overlap = 0.0
        for bbox_idx, bbox_mask in enumerate(merged_bbox_masks):
            bbox_seg = np.asarray(bbox_mask["segmentation"], dtype=bool)
            bbox_area = int(bbox_seg.sum())
            if bbox_area == 0:
                continue

            intersection = int(np.logical_and(point_seg, bbox_seg).sum())
            overlap_on_small = intersection / float(min(point_area, bbox_area))
            if overlap_on_small > best_overlap:
                best_overlap = overlap_on_small
                best_bbox_idx = bbox_idx

        if best_bbox_idx > -1 and best_overlap >= overlap_th:
            bbox_seg = np.asarray(merged_bbox_masks[best_bbox_idx]["segmentation"], dtype=bool)
            merged_seg = np.logical_or(bbox_seg, point_seg)
            merged_bbox_masks[best_bbox_idx]["segmentation"] = merged_seg
            merged_bbox_masks[best_bbox_idx]["predicted_iou"] = max(
                float(merged_bbox_masks[best_bbox_idx].get("predicted_iou", 0.0)),
                float(point_mask.get("predicted_iou", 0.0)),
            )
            merged_bbox_masks[best_bbox_idx]["stability_score"] = max(
                float(merged_bbox_masks[best_bbox_idx].get("stability_score", 0.0)),
                float(point_mask.get("stability_score", 0.0)),
            )
        else:
            remaining_point_masks.append(point_mask)

    return merged_bbox_masks, remaining_point_masks


def _extract_point(mask: Dict[str, object]) -> Tuple[int, int] | None:
    point_coords = mask.get("point_coords")
    if not point_coords:
        return None
    x, y = point_coords[0]
    return int(round(x)), int(round(y))


def render_debug_views(
    image: np.ndarray,
    bbox_prompts: Sequence[Dict[str, np.ndarray | int]],
    point_masks_raw: Sequence[Dict[str, object]],
    point_masks_selected: Sequence[Dict[str, object]],
    point_masks_premerge: Sequence[Dict[str, object]],
    point_masks_used: Sequence[Dict[str, object]],
    bbox_masks_premerge: Sequence[Dict[str, object]],
    bbox_masks: Sequence[Dict[str, object]],
    final_seg_map: np.ndarray,
    uncovered_mask: np.ndarray | None = None,
    valid_uncovered_mask_: np.ndarray | None = None,
    initial_point_grid: np.ndarray | None = None,
    sam_input_point_grid: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    input_view = image.astype(np.uint8).copy()
    bbox_view = input_view.copy()
    point_grid_initial_view = input_view.copy()
    point_grid_sam_input_view = input_view.copy()
    point_raw_view = input_view.copy()
    point_selected_view = input_view.copy()
    point_used_view = input_view.copy()
    bbox_result = _blend_overlay(input_view, _seg_idx_to_rgb(_masks_to_segmap(bbox_masks_premerge, image.shape[:2])))
    point_result_premerge = _blend_overlay(input_view, _seg_idx_to_rgb(_masks_to_segmap(point_masks_premerge, image.shape[:2])))
    point_result = _blend_overlay(input_view, _seg_idx_to_rgb(_masks_to_segmap(point_masks_used, image.shape[:2])))
    final_result = _blend_overlay(input_view, _seg_idx_to_rgb(final_seg_map))
    uncovered_view = _binary_mask_to_rgb(uncovered_mask, image.shape[:2])
    valid_uncovered_view = _binary_mask_to_rgb(valid_uncovered_mask_, image.shape[:2])

    for prompt in bbox_prompts:
        _draw_box(bbox_view, np.asarray(prompt["box"], dtype=np.float32))

    if initial_point_grid is not None and np.asarray(initial_point_grid).size > 0:
        for point in np.asarray(initial_point_grid, dtype=np.float32):
            _draw_point(
                point_grid_initial_view,
                (int(round(float(point[0]))), int(round(float(point[1])))),
                color=(0, 0, 255),
            )
    if sam_input_point_grid is not None and np.asarray(sam_input_point_grid).size > 0:
        for point in np.asarray(sam_input_point_grid, dtype=np.float32):
            _draw_point(
                point_grid_sam_input_view,
                (int(round(float(point[0]))), int(round(float(point[1])))),
                color=(255, 0, 255),
            )

    for mask in point_masks_raw:
        point = _extract_point(mask)
        if point is not None:
            _draw_point(point_raw_view, point, color=(255, 0, 0))

    for mask in point_masks_selected:
        point = _extract_point(mask)
        if point is not None:
            _draw_point(point_selected_view, point)

    for mask in point_masks_used:
        point = _extract_point(mask)
        if point is not None:
            _draw_point(point_used_view, point, color=(255, 255, 0))

    return {
        "input": input_view,
        "bbox_prompts": bbox_view,
        "point_grid_initial": point_grid_initial_view,
        "point_grid_sam_input": point_grid_sam_input_view,
        "point_prompts_raw": point_raw_view,
        "point_prompts_selected": point_selected_view,
        "point_prompts_used": point_used_view,
        "point_result_premerge": point_result_premerge,
        "point_result": point_result,
        "bbox_result": bbox_result,
        "final_result": final_result,
        "uncovered": uncovered_view,
        "valid_uncovered": valid_uncovered_view,
    }


def _masks_to_segmap(masks: Sequence[Dict[str, object]], image_shape: Tuple[int, int]) -> np.ndarray:
    seg_map = -np.ones(image_shape, dtype=np.int32)
    for idx, mask in enumerate(masks):
        seg = np.asarray(mask["segmentation"], dtype=bool)
        seg_map[np.logical_and(seg, seg_map < 0)] = idx
    return seg_map


def _seg_idx_to_rgb(seg_map: np.ndarray) -> np.ndarray:
    colours = np.asarray(
        [
            (166, 206, 227),
            (31, 120, 180),
            (178, 223, 138),
            (51, 160, 44),
            (251, 154, 153),
            (227, 26, 28),
            (253, 191, 111),
            (255, 127, 0),
            (202, 178, 214),
            (106, 61, 154),
        ],
        dtype=np.uint8,
    )
    rgb = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    if seg_map.size == 0:
        return rgb
    for idx in range(int(seg_map.max()) + 1):
        rgb[seg_map == idx] = colours[idx % len(colours)]
    return rgb


def _blend_overlay(image: np.ndarray, seg_rgb: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    if seg_rgb.size == 0:
        return image.astype(np.uint8).copy()
    image_f = image.astype(np.float32)
    seg_f = seg_rgb.astype(np.float32)
    valid = np.any(seg_rgb > 0, axis=-1)
    blended = image_f.copy()
    blended[valid] = (1.0 - alpha) * image_f[valid] + alpha * seg_f[valid]
    return np.clip(blended, 0, 255).astype(np.uint8)


def _binary_mask_to_rgb(mask: np.ndarray | None, image_shape: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    if mask is None:
        return vis
    mask_bool = np.asarray(mask, dtype=bool)
    vis[mask_bool] = (255, 255, 255)
    return vis


def _refine_binary_mask(mask: np.ndarray, dilation_kernel: int, closing_kernel: int) -> np.ndarray:
    refined = np.asarray(mask, dtype=np.uint8)
    if dilation_kernel > 1:
        kernel = np.ones((dilation_kernel, dilation_kernel), dtype=np.uint8)
        refined = cv2.dilate(refined, kernel, iterations=1)
    if closing_kernel > 1:
        kernel = np.ones((closing_kernel, closing_kernel), dtype=np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
    return refined.astype(bool)


def _compute_interior_distance(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = np.asarray(mask, dtype=np.uint8)
    if mask_uint8.size == 0:
        return np.zeros_like(mask_uint8, dtype=np.float32)
    padded = np.pad(mask_uint8, pad_width=1, mode="constant", constant_values=0)
    dist = cv2.distanceTransform(padded, distanceType=cv2.DIST_L2, maskSize=5)[1:-1, 1:-1]
    if dist.size == 0:
        return dist
    return dist


def _sample_single_mask_point(mask: np.ndarray) -> np.ndarray | None:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return None
    dist = _compute_interior_distance(mask_bool)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    if dist[y, x] <= 0:
        interior_points = np.argwhere(mask_bool)
        if interior_points.size == 0:
            return None
        y, x = interior_points[0]
    return np.array([float(x), float(y)], dtype=np.float32)


def _seen_unknown_sampling_policy(
    *,
    component_mask: np.ndarray,
    area: int,
    cell_area: float,
    cell_size_px: float,
) -> Tuple[int, int]:
    safe_cell_area = max(cell_area, 1.0)
    area_in_cells = area / (4.0 * safe_cell_area)
    n_points = max(1, int(np.ceil(area_in_cells)))

    dist = _compute_interior_distance(np.asarray(component_mask, dtype=bool))
    max_dist = float(dist.max()) if dist.size > 0 else 0.0
    divisor = min(max(n_points, 1), 5)
    margin_px = max(1, int(np.round(max_dist / float(divisor))))
    return n_points, margin_px


def _sample_inward_contour_points(
    component_mask: np.ndarray,
    *,
    n_points: int,
    margin_px: int,
) -> np.ndarray:
    mask_bool = np.asarray(component_mask, dtype=bool)
    if mask_bool.size == 0 or mask_bool.sum() == 0 or n_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n_points == 1:
        point = _sample_single_mask_point(mask_bool)
        if point is None:
            return np.zeros((0, 2), dtype=np.float32)
        return point.reshape(1, 2).astype(np.float32)

    dist = _compute_interior_distance(mask_bool)
    if dist.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    dist_values = dist[ys, xs]
    max_dist = float(dist_values.max())
    if max_dist <= 0.0:
        return np.zeros((0, 2), dtype=np.float32)

    step = max_dist / float(n_points)
    target_levels = [step * float(i) for i in range(n_points, 0, -1)]
    selected_points: List[np.ndarray] = []
    used_coords: set[tuple[int, int]] = set()
    for target in target_levels:
        order = np.argsort(np.abs(dist_values - target))
        chosen = None
        for idx in order:
            x = int(xs[idx])
            y = int(ys[idx])
            coord = (x, y)
            if coord in used_coords:
                continue
            chosen = np.array([float(x), float(y)], dtype=np.float32)
            used_coords.add(coord)
            break
        if chosen is not None:
            selected_points.append(chosen)

    if not selected_points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack(selected_points, axis=0)


def _render_selective_prompt_views(
    image: np.ndarray,
    known_masks: Dict[int, np.ndarray],
    known_union_mask: np.ndarray,
    known_points: Dict[int, np.ndarray],
    seen_unknown_mask: np.ndarray,
    seen_unknown_points: np.ndarray,
    brand_new_unknown_mask: np.ndarray,
    unknown_mask: np.ndarray,
    unknown_regions: Sequence[Dict[str, object]],
    sam_surviving_points: np.ndarray | None = None,
    sam_seg_map: np.ndarray | None = None,
    projected_object_ids_view: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    input_view = image.astype(np.uint8).copy()
    all_points_view = input_view.copy()
    sam_surviving_points_view = input_view.copy()
    projected_object_ids_panel = (
        input_view.copy()
        if projected_object_ids_view is None
        else np.asarray(projected_object_ids_view, dtype=np.uint8).copy()
    )
    sam_result_view = _blend_overlay(
        input_view,
        _seg_idx_to_rgb(
            -np.ones(image.shape[:2], dtype=np.int32)
            if sam_seg_map is None or np.asarray(sam_seg_map).size == 0
            else np.asarray(sam_seg_map, dtype=np.int32)
        ),
    )

    known_union_view = _blend_overlay(input_view, _binary_mask_to_rgb(known_union_mask, image.shape[:2]))
    seen_unknown_view = _blend_overlay(input_view, _binary_mask_to_rgb(seen_unknown_mask, image.shape[:2]))
    brand_new_unknown_view = _blend_overlay(input_view, _binary_mask_to_rgb(brand_new_unknown_mask, image.shape[:2]))
    _ = known_union_view, seen_unknown_view, brand_new_unknown_view, unknown_mask
    for ins_id in sorted(known_masks.keys()):
        for point in np.asarray(known_points.get(ins_id, np.zeros((0, 2), dtype=np.float32)), dtype=np.float32):
            pt = (int(round(float(point[0]))), int(round(float(point[1]))))
            _draw_point(all_points_view, pt, color=(255, 0, 0))

    for point in np.asarray(seen_unknown_points, dtype=np.float32):
        pt = (int(round(float(point[0]))), int(round(float(point[1]))))
        _draw_point(all_points_view, pt, color=(0, 255, 255))

    for region in unknown_regions:
        for point in np.asarray(region.get("points", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32):
            pt = (int(round(float(point[0]))), int(round(float(point[1]))))
            _draw_point(all_points_view, pt, color=(255, 255, 0))

    for point in np.asarray(
        np.zeros((0, 2), dtype=np.float32) if sam_surviving_points is None else sam_surviving_points,
        dtype=np.float32,
    ):
        pt = (int(round(float(point[0]))), int(round(float(point[1]))))
        _draw_point(sam_surviving_points_view, pt, color=(0, 255, 0))

    summary = np.concatenate(
        [
            np.concatenate((all_points_view, sam_surviving_points_view), axis=1),
            np.concatenate((projected_object_ids_panel, sam_result_view), axis=1),
        ],
        axis=0,
    )
    return {"summary": summary}


def _draw_box(image: np.ndarray, box: np.ndarray, color: Tuple[int, int, int] = (255, 128, 0)) -> None:
    if image.size == 0:
        return
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    image[y1, x1 : x2 + 1] = color
    image[y2, x1 : x2 + 1] = color
    image[y1 : y2 + 1, x1] = color
    image[y1 : y2 + 1, x2] = color


def _draw_point(image: np.ndarray, point: Tuple[int, int], color: Tuple[int, int, int] = (0, 255, 0), radius: int = 1) -> None:
    if image.size == 0:
        return
    h, w = image.shape[:2]
    x, y = point
    for yy in range(max(0, y - radius), min(h, y + radius + 1)):
        for xx in range(max(0, x - radius), min(w, x + radius + 1)):
            image[yy, xx] = color


def _render_seen_unknown_component_debug_view(
    image: np.ndarray,
    component_info: Dict[str, object],
) -> np.ndarray:
    component_mask = np.asarray(component_info.get("mask", np.zeros(image.shape[:2], dtype=bool)), dtype=bool)
    skipped = bool(component_info.get("skipped", False))
    if skipped:
        view = image.astype(np.uint8).copy()
        tint = np.zeros_like(view)
        tint[component_mask] = (255, 64, 64)
        view = _blend_overlay(view, tint, alpha=0.5)
    else:
        dist = _compute_interior_distance(component_mask)
        if dist.size == 0 or float(dist.max()) <= 0.0:
            view = image.astype(np.uint8).copy()
        else:
            dist_norm = np.clip(dist / float(dist.max()), 0.0, 1.0)
            dist_uint8 = np.round(dist_norm * 255.0).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(dist_uint8, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
            heatmap[~component_mask] = 0
            view = _blend_overlay(image.astype(np.uint8).copy(), heatmap, alpha=0.75)

    ys, xs = np.nonzero(component_mask)
    if len(xs) > 0:
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        _draw_box(view, np.array([x1, y1, x2, y2], dtype=np.float32), color=(255, 255, 255))

    for point in np.asarray(component_info.get("points", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32):
        pt = (int(round(float(point[0]))), int(round(float(point[1]))))
        _draw_point(view, pt, color=(255, 255, 0), radius=2)

    status_text = "SKIPPED" if skipped else "KEPT"
    if skipped:
        detail = str(component_info.get("skip_reason", ""))
    else:
        detail = (
            f"n={int(component_info.get('n_points', 0))} "
            f"m={int(component_info.get('margin_px', 0))} "
            f"maxd={float(component_info.get('max_dist', 0.0)):.2f}"
        )
    label = f"{status_text} area={int(component_info.get('area', 0))} {detail}".strip()
    cv2.putText(view, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return view
