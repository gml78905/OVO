from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json

import numpy as np


@dataclass
class InstancePrediction:
    scene: str
    class_id: int
    score: float
    face_mask: np.ndarray


@dataclass
class InstanceGroundTruth:
    scene: str
    class_id: int
    object_id: int
    face_mask: np.ndarray


def rle_decode(rle: Dict[str, str]) -> np.ndarray:
    length = rle["length"]
    counts = rle["counts"]
    s = counts.split()
    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def face_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def load_replica_gt_instances(
    replica_scene_dir: Path,
    map_to_reduced: Dict[int, int],
) -> List[InstanceGroundTruth]:
    from plyfile import PlyData

    habitat_dir = replica_scene_dir / "habitat"
    ply = PlyData.read(str(habitat_dir / "mesh_semantic.ply"))
    face_object_ids = np.asarray(ply["face"]["object_id"], dtype=np.int32)

    info = json.loads((habitat_dir / "info_semantic.json").read_text())
    gt_instances: List[InstanceGroundTruth] = []
    for obj in info["objects"]:
        raw_class_id = int(obj["class_id"])
        object_id = int(obj["id"])
        reduced_class_id = map_to_reduced.get(raw_class_id, -1)
        if reduced_class_id < 0:
            continue
        face_mask = face_object_ids == object_id
        if not np.any(face_mask):
            continue
        gt_instances.append(
            InstanceGroundTruth(
                scene=replica_scene_dir.name.replace("_", ""),
                class_id=reduced_class_id,
                object_id=object_id,
                face_mask=face_mask,
            )
        )
    return gt_instances


def load_replica_pred_instances(
    pred_dir: Path,
    scene: str,
    faces: np.ndarray,
) -> List[InstancePrediction]:
    pred_file = pred_dir / f"{scene}.txt"
    predictions: List[InstancePrediction] = []
    for line in pred_file.read_text().splitlines():
        rel_mask_path, class_id_str, score_str = line.split()
        vertex_mask = rle_decode(json.loads((pred_dir / rel_mask_path).read_text())).astype(bool)
        face_mask = vertex_mask[faces].sum(axis=1) >= 2
        if not np.any(face_mask):
            continue
        predictions.append(
            InstancePrediction(
                scene=scene,
                class_id=int(class_id_str),
                score=float(score_str),
                face_mask=face_mask,
            )
        )
    return predictions


def load_replica_faces(replica_scene_dir: Path) -> np.ndarray:
    from plyfile import PlyData

    ply = PlyData.read(str(replica_scene_dir / "habitat" / "mesh_semantic.ply"))
    faces = np.vstack(ply["face"]["vertex_indices"]).astype(np.int64)
    return faces


def evaluate_ap50(
    predictions: Iterable[InstancePrediction],
    ground_truths: Iterable[InstanceGroundTruth],
    iou_threshold: float = 0.5,
) -> Tuple[Dict[int, float], float]:
    preds_by_class: Dict[int, List[InstancePrediction]] = {}
    gts_by_class: Dict[int, List[InstanceGroundTruth]] = {}
    for pred in predictions:
        preds_by_class.setdefault(pred.class_id, []).append(pred)
    for gt in ground_truths:
        gts_by_class.setdefault(gt.class_id, []).append(gt)

    per_class_ap: Dict[int, float] = {}
    valid_aps: List[float] = []
    for class_id, gt_list in gts_by_class.items():
        preds = sorted(preds_by_class.get(class_id, []), key=lambda x: x.score, reverse=True)
        if not gt_list:
            continue

        matched = np.zeros(len(gt_list), dtype=bool)
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for i, pred in enumerate(preds):
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_list):
                if matched[gt_idx] or pred.scene != gt.scene:
                    continue
                iou = face_iou(pred.face_mask, gt.face_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched[best_gt_idx] = True
                tp[i] = 1.0
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / max(len(gt_list), 1)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)
        ap = compute_ap(recalls, precisions)
        per_class_ap[class_id] = ap
        valid_aps.append(ap)

    map50 = float(np.mean(valid_aps)) if valid_aps else float("nan")
    return per_class_ap, map50


def evaluate_ap(
    predictions: Iterable[InstancePrediction],
    ground_truths: Iterable[InstanceGroundTruth],
    iou_threshold: float,
) -> Tuple[Dict[int, float], float]:
    return evaluate_ap50(predictions, ground_truths, iou_threshold=iou_threshold)


def evaluate_instance_miou(
    predictions: Iterable[InstancePrediction],
    ground_truths: Iterable[InstanceGroundTruth],
) -> float:
    pred_list = list(predictions)
    gt_list = list(ground_truths)
    if not gt_list:
        return float("nan")

    best_ious: List[float] = []
    for gt in gt_list:
        same_class_preds = [
            pred
            for pred in pred_list
            if pred.scene == gt.scene and pred.class_id == gt.class_id
        ]
        if same_class_preds:
            best_ious.append(max(face_iou(pred.face_mask, gt.face_mask) for pred in same_class_preds))
        else:
            best_ious.append(0.0)
    return float(np.mean(best_ious))
