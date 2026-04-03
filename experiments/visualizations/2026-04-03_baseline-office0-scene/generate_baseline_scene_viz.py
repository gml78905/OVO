from __future__ import annotations

from pathlib import Path
import math
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import yaml

REPO_ROOT = Path('/ws/external')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ovo.utils import path_utils
from run_eval import load_representation


SCENE_PATH = Path('/ws/data/OVO/output/Replica/baseline_office0/office0')
EXPERIMENT_ROOT = Path('/ws/external/experiments/visualizations/2026-04-03_baseline-office0-scene')
POSE_PATH = SCENE_PATH / 'estimated_c2w.npy'
MAP_PATH = SCENE_PATH / 'ovo_map.ckpt'
RGB_ROOT = Path('/ws/data/OVO/input/Datasets/Replica/office0/results')
SAMPLE_FRAMES = [0, 400, 800, 1200, 1600, 1990]
POINT_SUBSAMPLE = 220_000
INSTANCE_SUBSAMPLE = 180_000
SEMANTIC_SUBSAMPLE = 180_000
SEED = 7


def build_palette(n: int) -> np.ndarray:
    colors = []
    cmaps = ['tab20', 'tab20b', 'tab20c', 'gist_ncar', 'hsv']
    for cmap_name in cmaps:
        cmap = plt.get_cmap(cmap_name)
        samples = 256 if cmap_name in ('gist_ncar', 'hsv') else cmap.N
        for i in range(samples):
            colors.append(cmap(i / max(samples - 1, 1))[:3])
    palette = np.asarray(colors, dtype=np.float32)
    if len(palette) < n:
        reps = math.ceil(n / len(palette))
        palette = np.tile(palette, (reps, 1))
    return palette[:n]


def alpha_blend(base: np.ndarray, overlay: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha = alpha[..., None].clip(0.0, 1.0)
    return (base * (1.0 - alpha) + overlay * alpha).astype(np.uint8)


def load_scene_assets():
    config = yaml.safe_load((SCENE_PATH / 'config.yaml').read_text())
    dataset_info = yaml.safe_load((path_utils.get_configs_root() / 'Replica' / 'eval_info.yaml').read_text())
    classes = dataset_info['class_names_reduced'] if dataset_info.get('map_to_reduced') else dataset_info['class_names']

    ovo, map_params = load_representation(SCENE_PATH, eval=True)
    instance_info = ovo.classify_instances(classes)
    object_ids = list(ovo.objects.keys())
    object_to_class = {int(obj_id): int(cls_idx) for obj_id, cls_idx in zip(object_ids, instance_info['classes'])}
    object_to_conf = {int(obj_id): float(conf) for obj_id, conf in zip(object_ids, instance_info['conf'])}

    xyz = map_params['xyz'].detach().cpu().numpy()
    rgb = map_params['color'].detach().cpu().numpy().astype(np.uint8)
    obj_ids = map_params['obj_ids'].detach().cpu().numpy().reshape(-1).astype(np.int64)
    poses = torch.load(POSE_PATH, map_location='cpu')

    semantic_ids = np.full(obj_ids.shape, -1, dtype=np.int64)
    valid_obj_mask = obj_ids >= 0
    semantic_ids[valid_obj_mask] = np.array([object_to_class.get(int(obj_id), -1) for obj_id in obj_ids[valid_obj_mask]], dtype=np.int64)

    return {
        'config': config,
        'classes': classes,
        'xyz': xyz,
        'rgb': rgb,
        'obj_ids': obj_ids,
        'semantic_ids': semantic_ids,
        'poses': poses,
        'object_to_class': object_to_class,
        'object_to_conf': object_to_conf,
    }


def subsample_indices(n: int, k: int, seed: int) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False))


def orthographic_views(points: np.ndarray, colors: np.ndarray, title: str, out_path: Path, labels: list[str]):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    scale = (maxs - mins).max()
    pts = (points - center) / max(scale, 1e-6)
    views = [
        (0, 1, 'Top-ish XY'),
        (0, 2, 'Front XZ'),
        (1, 2, 'Side YZ'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=180)
    for ax, (i, j, name) in zip(axes, views):
        order = np.argsort(pts[:, 2 - (i == 2) - (j == 2)])
        ax.scatter(pts[order, i], pts[order, j], c=colors[order], s=0.2, linewidths=0)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_facecolor('black')
    fig.suptitle(title, fontsize=16)
    if labels:
        fig.text(0.5, 0.03, ' | '.join(labels), ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def rasterize_projection(points_cam: np.ndarray, colors: np.ndarray, H: int, W: int, fx: float, fy: float, cx: float, cy: float):
    z = points_cam[:, 2]
    valid = z > 1e-4
    pts = points_cam[valid]
    cols = colors[valid]
    z = z[valid]
    u = np.round((pts[:, 0] * fx / z) + cx).astype(np.int32)
    v = np.round((pts[:, 1] * fy / z) + cy).astype(np.int32)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(inside):
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), dtype=np.float32)
    u = u[inside]
    v = v[inside]
    z = z[inside]
    cols = cols[inside]

    order = np.argsort(z)[::-1]
    u = u[order]
    v = v[order]
    cols = cols[order]

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    alpha = np.zeros((H, W), dtype=np.float32)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            uu = np.clip(u + du, 0, W - 1)
            vv = np.clip(v + dv, 0, H - 1)
            canvas[vv, uu] = cols
            alpha[vv, uu] = 0.85
    return canvas, alpha


def make_frame_grid(scene, out_path: Path):
    cfg = scene['config']
    H = int(cfg['cam']['H'])
    W = int(cfg['cam']['W'])
    fx = float(cfg['cam']['fx'])
    fy = float(cfg['cam']['fy'])
    cx = float(cfg['cam']['cx'])
    cy = float(cfg['cam']['cy'])

    xyz = scene['xyz']
    rgb_points = scene['rgb']
    obj_ids = scene['obj_ids']
    sem_ids = scene['semantic_ids']
    valid_obj = obj_ids >= 0
    valid_sem = sem_ids >= 0

    obj_unique = np.sort(np.unique(obj_ids[valid_obj]))
    obj_palette = build_palette(len(obj_unique))
    obj_color_map = {int(obj_id): (obj_palette[i] * 255).astype(np.uint8) for i, obj_id in enumerate(obj_unique)}
    obj_colors = np.zeros_like(rgb_points)
    if np.any(valid_obj):
        obj_colors[valid_obj] = np.array([obj_color_map[int(obj_id)] for obj_id in obj_ids[valid_obj]], dtype=np.uint8)

    sem_unique = np.sort(np.unique(sem_ids[valid_sem]))
    sem_palette = build_palette(len(sem_unique))
    sem_color_map = {int(sem_id): (sem_palette[i] * 255).astype(np.uint8) for i, sem_id in enumerate(sem_unique)}
    sem_colors = np.zeros_like(rgb_points)
    if np.any(valid_sem):
        sem_colors[valid_sem] = np.array([sem_color_map[int(sem_id)] for sem_id in sem_ids[valid_sem]], dtype=np.uint8)

    rgb_idx = subsample_indices(len(xyz), POINT_SUBSAMPLE, SEED)
    obj_idx = subsample_indices(np.where(valid_obj)[0].shape[0], INSTANCE_SUBSAMPLE, SEED + 1)
    sem_idx = subsample_indices(np.where(valid_sem)[0].shape[0], SEMANTIC_SUBSAMPLE, SEED + 2)
    obj_points_idx = np.where(valid_obj)[0][obj_idx] if np.any(valid_obj) else np.array([], dtype=np.int64)
    sem_points_idx = np.where(valid_sem)[0][sem_idx] if np.any(valid_sem) else np.array([], dtype=np.int64)

    rows = len(SAMPLE_FRAMES)
    fig, axes = plt.subplots(rows, 4, figsize=(20, 4 * rows), dpi=150)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, frame_id in enumerate(SAMPLE_FRAMES):
        rgb_path = RGB_ROOT / f'frame{frame_id:06d}.jpg'
        base = np.array(Image.open(rgb_path).convert('RGB'))
        c2w = np.asarray(scene['poses'][frame_id], dtype=np.float32)
        w2c = np.linalg.inv(c2w)

        pts_h = np.concatenate([xyz[rgb_idx], np.ones((len(rgb_idx), 1), dtype=np.float32)], axis=1)
        points_cam = (w2c @ pts_h.T).T[:, :3]
        rgb_canvas, rgb_alpha = rasterize_projection(points_cam, rgb_points[rgb_idx], H, W, fx, fy, cx, cy)
        rgb_overlay = alpha_blend(base, rgb_canvas, rgb_alpha * 0.65)

        obj_overlay = base.copy()
        if len(obj_points_idx) > 0:
            obj_h = np.concatenate([xyz[obj_points_idx], np.ones((len(obj_points_idx), 1), dtype=np.float32)], axis=1)
            obj_cam = (w2c @ obj_h.T).T[:, :3]
            obj_canvas, obj_alpha = rasterize_projection(obj_cam, obj_colors[obj_points_idx], H, W, fx, fy, cx, cy)
            obj_overlay = alpha_blend(base, obj_canvas, obj_alpha)

        sem_overlay = base.copy()
        if len(sem_points_idx) > 0:
            sem_h = np.concatenate([xyz[sem_points_idx], np.ones((len(sem_points_idx), 1), dtype=np.float32)], axis=1)
            sem_cam = (w2c @ sem_h.T).T[:, :3]
            sem_canvas, sem_alpha = rasterize_projection(sem_cam, sem_colors[sem_points_idx], H, W, fx, fy, cx, cy)
            sem_overlay = alpha_blend(base, sem_canvas, sem_alpha)

        axes[row, 0].imshow(base)
        axes[row, 1].imshow(rgb_overlay)
        axes[row, 2].imshow(obj_overlay)
        axes[row, 3].imshow(sem_overlay)
        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(f'frame {frame_id}', fontsize=11)

    for ax, title in zip(axes[0], ['RGB', 'Projected RGB Map', 'Instance Overlay', 'Semantic Overlay']):
        ax.set_title(title, fontsize=13)
    fig.suptitle('Baseline office0 Scene Projections', fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

    return {
        'object_count': int(len(obj_unique)),
        'semantic_count': int(len(sem_unique)),
        'frames': SAMPLE_FRAMES,
    }


def make_3d_views(scene):
    xyz = scene['xyz']
    rgb = scene['rgb'].astype(np.float32) / 255.0
    obj_ids = scene['obj_ids']
    sem_ids = scene['semantic_ids']

    valid_obj = obj_ids >= 0
    valid_sem = sem_ids >= 0

    all_idx = subsample_indices(len(xyz), POINT_SUBSAMPLE, SEED)
    orthographic_views(xyz[all_idx], rgb[all_idx], 'Baseline office0: RGB-colored map points', EXPERIMENT_ROOT / 'map_rgb_views.png', [])

    obj_unique = np.sort(np.unique(obj_ids[valid_obj]))
    obj_palette = build_palette(len(obj_unique))
    obj_map = {int(obj_id): obj_palette[i] for i, obj_id in enumerate(obj_unique)}
    obj_idx = np.where(valid_obj)[0]
    obj_idx = obj_idx[subsample_indices(len(obj_idx), INSTANCE_SUBSAMPLE, SEED + 3)]
    obj_colors = np.array([obj_map[int(obj_id)] for obj_id in obj_ids[obj_idx]], dtype=np.float32)
    orthographic_views(xyz[obj_idx], obj_colors, 'Baseline office0: instance-colored map points', EXPERIMENT_ROOT / 'map_instance_views.png', [f'instances={len(obj_unique)}'])

    sem_unique = np.sort(np.unique(sem_ids[valid_sem]))
    sem_palette = build_palette(len(sem_unique))
    sem_map = {int(sem_id): sem_palette[i] for i, sem_id in enumerate(sem_unique)}
    sem_idx = np.where(valid_sem)[0]
    sem_idx = sem_idx[subsample_indices(len(sem_idx), SEMANTIC_SUBSAMPLE, SEED + 4)]
    sem_colors = np.array([sem_map[int(sem_id)] for sem_id in sem_ids[sem_idx]], dtype=np.float32)
    top_classes = []
    uniq, counts = np.unique(sem_ids[valid_sem], return_counts=True)
    order = np.argsort(counts)[::-1][:8]
    top_classes = [f"{scene['classes'][int(uniq[i])]}={int(counts[i])}" for i in order]
    orthographic_views(xyz[sem_idx], sem_colors, 'Baseline office0: semantic-colored map points', EXPERIMENT_ROOT / 'map_semantic_views.png', top_classes)


def save_summary(scene, frame_meta):
    valid_obj = scene['obj_ids'] >= 0
    valid_sem = scene['semantic_ids'] >= 0
    uniq_obj, obj_counts = np.unique(scene['obj_ids'][valid_obj], return_counts=True)
    uniq_sem, sem_counts = np.unique(scene['semantic_ids'][valid_sem], return_counts=True)
    top_objects = [
        {
            'object_id': int(obj_id),
            'point_count': int(count),
            'semantic_class': scene['classes'][scene['object_to_class'].get(int(obj_id), 0)],
            'semantic_confidence': round(scene['object_to_conf'].get(int(obj_id), 0.0), 4),
        }
        for obj_id, count in sorted(zip(uniq_obj, obj_counts), key=lambda x: x[1], reverse=True)[:15]
    ]
    top_semantics = [
        {
            'class_index': int(sem_id),
            'class_name': scene['classes'][int(sem_id)],
            'point_count': int(count),
        }
        for sem_id, count in sorted(zip(uniq_sem, sem_counts), key=lambda x: x[1], reverse=True)[:15]
    ]
    summary = {
        'scene': 'Replica/office0',
        'run': 'baseline_office0',
        'map_point_count': int(scene['xyz'].shape[0]),
        'instance_point_count': int(valid_obj.sum()),
        'semantic_point_count': int(valid_sem.sum()),
        'instance_count': int(len(uniq_obj)),
        'semantic_class_count': int(len(uniq_sem)),
        'sample_frames': frame_meta['frames'],
        'top_objects': top_objects,
        'top_semantics': top_semantics,
    }
    (EXPERIMENT_ROOT / 'summary.json').write_text(json.dumps(summary, indent=2))


def write_readme():
    text = """# Baseline office0 Scene Visualization

This folder contains qualitative visualizations for the `baseline_office0` experiment only.

Files:
- `scene_frame_overlays.png`: representative RGB frames with projected baseline map overlays.
- `map_rgb_views.png`: multi-view render of the reconstructed map colored by original RGB.
- `map_instance_views.png`: multi-view render colored by baseline instance ids.
- `map_semantic_views.png`: multi-view render colored by baseline semantic predictions.
- `summary.json`: compact metadata for the generated visuals.

Notes:
- Semantic labels are restored from the baseline checkpoint and recomputed with the saved CLIP descriptors.
- Frame overlays are generated by projecting reconstructed 3D map points into selected camera views.
- This is a qualitative visualization for inspection, not the mesh-level evaluation artifact.
"""
    (EXPERIMENT_ROOT / 'README.md').write_text(text)


def main():
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    scene = load_scene_assets()
    make_3d_views(scene)
    frame_meta = make_frame_grid(scene, EXPERIMENT_ROOT / 'scene_frame_overlays.png')
    save_summary(scene, frame_meta)
    write_readme()
    print(f'Saved baseline scene visualization to {EXPERIMENT_ROOT}')


if __name__ == '__main__':
    main()
