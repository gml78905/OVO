from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


class ProbabilisticGrouping:
    """Keeps reversible object-to-object merge evidence and dynamic semantic groups.

    The low-level tracked instances remain as conservative proto-objects. This class
    accumulates ambiguous cross-frame evidence between proto-objects, builds a dynamic
    grouping graph from that evidence, and exposes group-level CLIP descriptors without
    irreversibly fusing the underlying objects.
    """

    def __init__(self, config: Dict | None = None) -> None:
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.commit_confidence = float(config.get("commit_confidence", 0.65))
        self.commit_margin = float(config.get("commit_margin", 0.15))
        self.min_candidate_prob = float(config.get("min_candidate_prob", 0.12))
        self.pos_obs_weight = float(config.get("pos_obs_weight", 1.0))
        self.neg_obs_weight = float(config.get("neg_obs_weight", 0.35))
        self.edge_posterior_th = float(config.get("edge_posterior_th", 0.72))
        self.clip_cossim_th = float(config.get("clip_cossim_th", 0.84))
        self.centroid_th = float(config.get("centroid_th", 1.5))
        self.decay = float(config.get("decay", 0.995))
        self.max_posterior_candidates = int(config.get("max_posterior_candidates", 3))
        self.point_belief_decay = float(config.get("point_belief_decay", 0.80))
        self.point_belief_top_k = int(config.get("point_belief_top_k", 3))
        self.allow_reassign = bool(config.get("allow_reassign", False))
        self.reassign_confidence = float(config.get("reassign_confidence", 0.80))
        self.reassign_margin = float(config.get("reassign_margin", 0.25))
        self.object_support_weight = float(config.get("object_support_weight", 0.25))
        self.object_support_neg_weight = float(config.get("object_support_neg_weight", 0.05))

        self.pairwise_evidence: Dict[Tuple[int, int], List[float]] = {}
        self.point_beliefs: Dict[int, Dict[int, float]] = {}
        self.object_groups: Dict[int, int] = {}
        self.group_members: Dict[int, List[int]] = {}
        self.group_clips: Dict[int, torch.Tensor] = {}
        self.last_group_summary = {
            "semantic_groups": 0,
            "grouped_proto_objects": 0,
            "multi_member_groups": 0,
            "active_pairs": 0,
            "max_pair_posterior": 0.0,
        }
        self.stats = {
            "frames_observed": 0,
            "ambiguous_segments": 0,
            "conservative_skips": 0,
            "committed_segments": 0,
            "reassigned_points": 0,
        }
        self._last_decay_kf = None

    @staticmethod
    def _pair_key(id1: int, id2: int) -> Tuple[int, int]:
        return (id1, id2) if id1 < id2 else (id2, id1)

    @staticmethod
    def _flatten_clip(feature: torch.Tensor | None) -> torch.Tensor | None:
        if feature is None:
            return None
        if feature.ndim == 1:
            return feature
        return feature.squeeze(0)

    @staticmethod
    def _restore_shape(feature: torch.Tensor, reference: torch.Tensor | None) -> torch.Tensor:
        if reference is not None and reference.ndim > 1:
            return feature.unsqueeze(0)
        return feature

    def decay_evidence(self, kf_id: int) -> None:
        if self._last_decay_kf == kf_id:
            return
        self._last_decay_kf = kf_id
        self.stats["frames_observed"] += 1
        if self.decay >= 0.9999:
            return
        keys_to_delete = []
        for key, values in self.pairwise_evidence.items():
            values[0] *= self.decay
            values[1] *= self.decay
            if values[0] + values[1] < 1e-4:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            self.pairwise_evidence.pop(key, None)

    def posterior(self, id1: int, id2: int) -> float:
        pos, neg = self.pairwise_evidence.get(self._pair_key(id1, id2), [0.0, 0.0])
        return (1.0 + pos) / (2.0 + pos + neg)

    def should_commit(self, candidate_scores: Dict[int, float]) -> tuple[bool, int, float, float]:
        if len(candidate_scores) == 0:
            return False, -1, 0.0, 0.0
        ordered = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        top_id, top_conf = ordered[0]
        second_conf = ordered[1][1] if len(ordered) > 1 else 0.0
        margin = top_conf - second_conf
        should_commit = top_conf >= self.commit_confidence and margin >= self.commit_margin
        if not should_commit:
            self.stats["conservative_skips"] += 1
        return should_commit, top_id, top_conf, margin

    def _normalize_sparse(self, scores: Dict[int, float]) -> Dict[int, float]:
        if len(scores) == 0:
            return {}
        items = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: self.point_belief_top_k]
        total = sum(max(value, 0.0) for _, value in items)
        if total <= 0:
            return {}
        return {int(key): float(value / total) for key, value in items}

    def _update_point_beliefs(self, point_ids: Iterable[int], posterior: Dict[int, float]) -> None:
        if len(posterior) <= 1:
            return
        sparse_posterior = self._normalize_sparse(posterior)
        if len(sparse_posterior) <= 1:
            return
        for point_id in point_ids:
            point_id = int(point_id)
            belief = self.point_beliefs.get(point_id, {})
            updated = {key: value * self.point_belief_decay for key, value in belief.items()}
            for ins_id, prob in sparse_posterior.items():
                updated[ins_id] = updated.get(ins_id, 0.0) + prob
            updated = self._normalize_sparse(updated)
            if len(updated) > 1:
                self.point_beliefs[point_id] = updated
            else:
                self.point_beliefs.pop(point_id, None)

    def reassign_points(
        self,
        point_ids: Iterable[int],
        current_point_ins_ids: torch.Tensor,
        candidate_scores: Dict[int, float],
    ) -> int:
        if not self.allow_reassign or len(candidate_scores) == 0:
            return 0
        ordered = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        top_id, top_conf = ordered[0]
        second_conf = ordered[1][1] if len(ordered) > 1 else 0.0
        if top_conf < self.reassign_confidence or top_conf - second_conf < self.reassign_margin:
            return 0

        reassigned = 0
        point_ids = list(point_ids)
        for local_idx, point_id in enumerate(point_ids):
            belief = self.point_beliefs.get(int(point_id))
            if belief is None:
                continue
            best_id = max(belief, key=belief.get)
            best_conf = belief[best_id]
            alt_conf = max([value for key, value in belief.items() if key != best_id], default=0.0)
            if best_id == top_id and best_conf >= self.reassign_confidence and best_conf - alt_conf >= self.reassign_margin:
                if current_point_ins_ids[local_idx].item() != best_id:
                    current_point_ins_ids[local_idx] = best_id
                    reassigned += 1
        self.stats["reassigned_points"] += reassigned
        return reassigned

    def observe_segments(self, kf_id: int, frame_segments: List[Dict]) -> None:
        if not self.enabled:
            return
        self.decay_evidence(kf_id)
        if len(frame_segments) == 0:
            return

        committed_conf = {}
        for segment in frame_segments:
            posterior = {
                int(ins_id): float(prob)
                for ins_id, prob in segment.get("posterior", {}).items()
                if float(prob) >= self.min_candidate_prob
            }
            if len(posterior) > self.max_posterior_candidates:
                posterior = dict(sorted(posterior.items(), key=lambda item: item[1], reverse=True)[: self.max_posterior_candidates])

            point_ids = segment.get("point_ids", [])
            if len(posterior) > 1:
                self.stats["ambiguous_segments"] += 1
                self._update_point_beliefs(point_ids, posterior)
                obs_weight = min(1.0, segment.get("point_count", 0) / max(segment.get("track_th", 1.0), 1.0))
                for (id1, prob1), (id2, prob2) in combinations(posterior.items(), 2):
                    key = self._pair_key(id1, id2)
                    evidence = self.pairwise_evidence.setdefault(key, [0.0, 0.0])
                    evidence[0] += self.pos_obs_weight * obs_weight * prob1 * prob2

            committed_id = int(segment.get("committed_id", -1))
            committed_confidence = float(segment.get("committed_confidence", 0.0))
            if committed_id > -1:
                self.stats["committed_segments"] += 1
                committed_conf[committed_id] = max(committed_conf.get(committed_id, 0.0), committed_confidence)

        for (id1, conf1), (id2, conf2) in combinations(committed_conf.items(), 2):
            key = self._pair_key(id1, id2)
            evidence = self.pairwise_evidence.setdefault(key, [0.0, 0.0])
            evidence[1] += self.neg_obs_weight * conf1 * conf2

    def _connected_components(self, nodes: List[int], edges: List[Tuple[int, int]]) -> Dict[int, int]:
        parent = {node: node for node in nodes}

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(node1: int, node2: int) -> None:
            root1 = find(node1)
            root2 = find(node2)
            if root1 == root2:
                return
            if root1 < root2:
                parent[root2] = root1
            else:
                parent[root1] = root2

        for node1, node2 in edges:
            union(node1, node2)

        groups = {}
        for node in nodes:
            groups[node] = find(node)
        return groups

    def update_groups(self, objects: Dict[int, object], obj_pcds: Dict[int, List[torch.Tensor]]) -> Dict[int, int]:
        if not self.enabled:
            self.object_groups = {obj_id: obj_id for obj_id in objects.keys()}
            self.group_members = {obj_id: [obj_id] for obj_id in objects.keys()}
            self.group_clips = {}
            self.last_group_summary = {
                "semantic_groups": len(self.object_groups),
                "grouped_proto_objects": 0,
                "multi_member_groups": 0,
                "active_pairs": 0,
                "max_pair_posterior": 0.0,
            }
            return self.object_groups

        nodes = sorted(objects.keys())
        active_edges = []
        active_pairs = 0
        max_pair_posterior = 0.0
        for id1, id2 in combinations(nodes, 2):
            obj1 = objects[id1]
            obj2 = objects[id2]
            clip1 = self._flatten_clip(getattr(obj1, "clip_feature", None))
            clip2 = self._flatten_clip(getattr(obj2, "clip_feature", None))
            if clip1 is None or clip2 is None:
                continue
            clip_sim = torch.nn.functional.cosine_similarity(clip1, clip2, dim=0).item()
            centroid1 = obj_pcds[id1][1]
            centroid2 = obj_pcds[id2][1]
            distance = ((centroid1 - centroid2) ** 2).sum().sqrt().item()
            if distance <= self.centroid_th and clip_sim >= self.clip_cossim_th:
                key = self._pair_key(id1, id2)
                evidence = self.pairwise_evidence.setdefault(key, [0.0, 0.0])
                clip_gain = (clip_sim - self.clip_cossim_th) / max(1e-6, 1.0 - self.clip_cossim_th)
                dist_gain = max(0.0, 1.0 - distance / max(self.centroid_th, 1e-6))
                evidence[0] += self.object_support_weight * min(1.0, 0.5 * (clip_gain + dist_gain))
            elif distance <= self.centroid_th and clip_sim < self.clip_cossim_th:
                key = self._pair_key(id1, id2)
                evidence = self.pairwise_evidence.setdefault(key, [0.0, 0.0])
                clip_gap = max(0.0, (self.clip_cossim_th - clip_sim) / max(self.clip_cossim_th, 1e-6))
                evidence[1] += self.object_support_neg_weight * min(1.0, clip_gap)

            posterior = self.posterior(id1, id2)
            if posterior < self.edge_posterior_th:
                continue
            if clip_sim < self.clip_cossim_th or distance > self.centroid_th:
                continue
            active_edges.append((id1, id2))
            active_pairs += 1
            max_pair_posterior = max(max_pair_posterior, posterior)

        self.object_groups = self._connected_components(nodes, active_edges)
        self.rebuild_group_clips(objects)

        self.last_group_summary = {
            "semantic_groups": len(self.group_members),
            "grouped_proto_objects": sum(len(members) for members in self.group_members.values() if len(members) > 1),
            "multi_member_groups": sum(1 for members in self.group_members.values() if len(members) > 1),
            "active_pairs": active_pairs,
            "max_pair_posterior": round(max_pair_posterior, 4),
        }
        return self.object_groups

    def rebuild_group_clips(self, objects: Dict[int, object]) -> None:
        self.group_members = {}
        for obj_id, group_id in self.object_groups.items():
            if obj_id in objects:
                self.group_members.setdefault(group_id, []).append(obj_id)
        self.group_members = {group_id: sorted(members) for group_id, members in self.group_members.items()}

        self.group_clips = {}
        for group_id, member_ids in self.group_members.items():
            member_clips = []
            member_weights = []
            reference = None
            for member_id in member_ids:
                clip_feature = getattr(objects[member_id], "clip_feature", None)
                clip_flat = self._flatten_clip(clip_feature)
                if clip_flat is None:
                    continue
                reference = clip_feature
                member_clips.append(clip_flat)
                member_weights.append(max(1.0, float(len(getattr(objects[member_id], "kfs_ids", [])))))
            if len(member_clips) == 0:
                continue
            if len(member_clips) == 1:
                group_clip = member_clips[0]
            else:
                clips = torch.stack(member_clips)
                weights = torch.tensor(member_weights, dtype=clips.dtype, device=clips.device)
                group_clip = (clips * weights[:, None]).sum(dim=0) / weights.sum()
                group_clip = torch.nn.functional.normalize(group_clip, p=2, dim=0)
            self.group_clips[group_id] = self._restore_shape(group_clip.cpu(), reference)

    def get_object_clip(self, ins_id: int, fallback: torch.Tensor) -> torch.Tensor:
        group_id = self.object_groups.get(ins_id)
        if group_id is None:
            return fallback
        return self.group_clips.get(group_id, fallback)

    def capture_dict(self) -> Dict[str, np.ndarray]:
        state: Dict[str, np.ndarray] = {
            "prob_grouping_enabled": np.array([int(self.enabled)], dtype=np.int8),
            "prob_group_stats": np.array(
                [
                    self.stats["frames_observed"],
                    self.stats["ambiguous_segments"],
                    self.stats["conservative_skips"],
                    self.stats["committed_segments"],
                    self.stats["reassigned_points"],
                ],
                dtype=np.float32,
            ),
        }
        if len(self.pairwise_evidence) > 0:
            pair_keys = np.array(sorted(self.pairwise_evidence.keys()), dtype=np.int64)
            pair_values = np.array([self.pairwise_evidence[key] for key in sorted(self.pairwise_evidence.keys())], dtype=np.float32)
            state["prob_group_pair_keys"] = pair_keys
            state["prob_group_pair_values"] = pair_values
        if len(self.object_groups) > 0:
            object_ids = np.array(sorted(self.object_groups.keys()), dtype=np.int64)
            group_ids = np.array([self.object_groups[obj_id] for obj_id in object_ids], dtype=np.int64)
            state["prob_group_object_ids"] = object_ids
            state["prob_group_group_ids"] = group_ids
        return state

    def restore_dict(self, scene_dict: Dict[str, np.ndarray]) -> None:
        enabled = scene_dict.get("prob_grouping_enabled")
        if enabled is not None:
            self.enabled = bool(int(enabled[0]))
        stats = scene_dict.get("prob_group_stats")
        if stats is not None and len(stats) >= 5:
            self.stats["frames_observed"] = int(stats[0])
            self.stats["ambiguous_segments"] = int(stats[1])
            self.stats["conservative_skips"] = int(stats[2])
            self.stats["committed_segments"] = int(stats[3])
            self.stats["reassigned_points"] = int(stats[4])

        self.pairwise_evidence = {}
        pair_keys = scene_dict.get("prob_group_pair_keys")
        pair_values = scene_dict.get("prob_group_pair_values")
        if pair_keys is not None and pair_values is not None:
            for key, value in zip(pair_keys, pair_values):
                self.pairwise_evidence[(int(key[0]), int(key[1]))] = [float(value[0]), float(value[1])]

        self.object_groups = {}
        object_ids = scene_dict.get("prob_group_object_ids")
        group_ids = scene_dict.get("prob_group_group_ids")
        if object_ids is not None and group_ids is not None:
            for obj_id, group_id in zip(object_ids, group_ids):
                self.object_groups[int(obj_id)] = int(group_id)
        self.group_members = {}
        for obj_id, group_id in self.object_groups.items():
            self.group_members.setdefault(group_id, []).append(obj_id)
        self.group_members = {group_id: sorted(members) for group_id, members in self.group_members.items()}
        self.group_clips = {}

    def summary(self) -> Dict[str, float]:
        summary = dict(self.last_group_summary)
        summary.update(self.stats)
        return summary
