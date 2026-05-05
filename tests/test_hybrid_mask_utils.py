import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from ovo.utils.hybrid_mask_utils import (
    build_known_object_masks,
    build_selective_prompt_plan,
    build_projected_bbox_prompts,
    collect_all_prompt_points,
    filter_point_grid_for_sam,
    filter_point_masks_by_coverage,
    merge_bbox_point_masks,
    render_debug_views,
    save_selective_prompt_debug_views,
    sample_mask_component_points,
    sample_seen_unknown_component_points,
    sample_unknown_region_points,
)


class HybridMaskUtilsTest(unittest.TestCase):
    def test_build_known_object_masks_groups_depth_consistent_points_by_instance(self):
        points_3d = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [4.0, 4.0, 1.0],
                [4.0, 5.0, 1.0],
            ],
            dtype=torch.float32,
        )
        points_ins_ids = torch.tensor([3, 3, 8, 8], dtype=torch.int64)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = torch.eye(3, dtype=torch.float32)
        c2w = torch.eye(4, dtype=torch.float32)

        known_masks, known_union = build_known_object_masks(
            points_3d,
            points_ins_ids,
            depth,
            intrinsics,
            c2w,
            image_shape=(8, 8),
            match_distance_th=0.05,
            min_points=2,
            dilation_kernel=1,
            closing_kernel=1,
        )

        self.assertEqual(sorted(known_masks.keys()), [3, 8])
        self.assertTrue(known_masks[3][1, 1])
        self.assertTrue(known_masks[3][2, 1])
        self.assertTrue(known_masks[8][4, 4])
        self.assertTrue(known_masks[8][5, 4])
        self.assertTrue(known_union[1, 1])
        self.assertTrue(known_union[5, 4])

    def test_build_known_object_masks_ignores_seen_but_unassigned_points(self):
        points_3d = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [4.0, 4.0, 1.0],
                [4.0, 5.0, 1.0],
            ],
            dtype=torch.float32,
        )
        points_ins_ids = torch.tensor([0, 0, 2, 2], dtype=torch.int64)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = torch.eye(3, dtype=torch.float32)
        c2w = torch.eye(4, dtype=torch.float32)

        known_masks, known_union = build_known_object_masks(
            points_3d,
            points_ins_ids,
            depth,
            intrinsics,
            c2w,
            image_shape=(8, 8),
            match_distance_th=0.05,
            min_points=2,
            dilation_kernel=1,
            closing_kernel=1,
        )

        self.assertEqual(sorted(known_masks.keys()), [2])
        self.assertFalse(known_union[1, 1])
        self.assertTrue(known_union[4, 4])

    def test_selective_prompt_plan_builds_known_component_points_and_unknown_points(self):
        points_3d = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [6.0, 1.0, 1.0],
                [6.0, 2.0, 1.0],
                [4.0, 4.0, 1.0],
                [4.0, 5.0, 1.0],
            ],
            dtype=torch.float32,
        )
        points_ins_ids = torch.tensor([3, 3, 3, 3, 8, 8], dtype=torch.int64)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = torch.eye(3, dtype=torch.float32)
        c2w = torch.eye(4, dtype=torch.float32)

        plan = build_selective_prompt_plan(
            points_3d,
            points_ins_ids,
            depth,
            intrinsics,
            c2w,
            image_shape=(8, 8),
            known_min_points=2,
            known_dilation_kernel=1,
            known_closing_kernel=1,
            unknown_min_area=4,
            unknown_area_per_point=8,
            unknown_max_points=4,
        )

        self.assertEqual(sorted(plan["known_masks"].keys()), [3, 8])
        self.assertEqual(sorted(plan["known_points"].keys()), [3, 8])
        self.assertEqual(plan["known_points"][3].shape[0], 2)
        self.assertEqual(plan["known_points"][8].shape[0], 1)
        self.assertTrue(np.any(plan["unknown_mask"]))
        self.assertGreaterEqual(len(plan["unknown_regions"]), 1)
        self.assertGreaterEqual(plan["unknown_regions"][0]["points"].shape[0], 1)

    def test_selective_prompt_plan_splits_known_seen_unknown_and_brand_new_unknown(self):
        points_3d = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [4.0, 4.0, 1.0],
                [4.0, 5.0, 1.0],
            ],
            dtype=torch.float32,
        )
        points_ins_ids = torch.tensor([0, 0, 2, 2], dtype=torch.int64)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = torch.eye(3, dtype=torch.float32)
        c2w = torch.eye(4, dtype=torch.float32)

        plan = build_selective_prompt_plan(
            points_3d,
            points_ins_ids,
            depth,
            intrinsics,
            c2w,
            image_shape=(8, 8),
            known_min_points=2,
            known_dilation_kernel=1,
            known_closing_kernel=1,
            unknown_min_area=4,
            unknown_area_per_point=8,
            unknown_max_points=4,
        )

        self.assertEqual(sorted(plan["known_masks"].keys()), [2])
        self.assertTrue(plan["seen_unknown_mask"][1, 1])
        self.assertTrue(plan["seen_unknown_mask"][2, 1])
        self.assertFalse(plan["brand_new_unknown_mask"][1, 1])
        self.assertFalse(plan["brand_new_unknown_mask"][4, 4])
        self.assertGreaterEqual(plan["seen_unknown_points"].shape[0], 1)
        self.assertTrue(np.any(plan["brand_new_unknown_mask"]))

    def test_sample_seen_unknown_component_points_uses_multiple_boundary_points_for_large_component(self):
        mask = np.zeros((24, 24), dtype=bool)
        mask[4:20, 4:20] = True

        sampled = sample_seen_unknown_component_points(
            mask,
            cell_size_px=8.0,
            min_area=16,
        )

        self.assertEqual(sampled.shape[0], 1)
        for point in sampled:
            self.assertTrue(mask[int(point[1]), int(point[0])])
        contour_dists = []
        for point in sampled:
            x = int(point[0])
            y = int(point[1])
            contour_dists.append(min(x - 4, 19 - x, y - 4, 19 - y))
        self.assertGreaterEqual(contour_dists[0], 7)

    def test_sample_seen_unknown_component_points_uses_max_distance_point_when_single_point(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[2:14, 2:14] = True

        sampled = sample_seen_unknown_component_points(
            mask,
            cell_size_px=24.0,
            min_area=16,
        )

        self.assertEqual(sampled.shape[0], 1)
        x = int(sampled[0][0])
        y = int(sampled[0][1])
        self.assertTrue(mask[y, x])
        self.assertGreaterEqual(min(x - 2, 13 - x, y - 2, 13 - y), 0)

    def test_sample_seen_unknown_component_points_treats_image_border_as_boundary_for_margin(self):
        mask = np.zeros((24, 24), dtype=bool)
        mask[0:16, 0:16] = True

        sampled = sample_seen_unknown_component_points(
            mask,
            cell_size_px=8.0,
            min_area=16,
        )

        self.assertEqual(sampled.shape[0], 1)
        for point in sampled:
            x = int(point[0])
            y = int(point[1])
            self.assertTrue(mask[y, x])
            self.assertGreater(x, 0)
            self.assertGreater(y, 0)

    def test_selective_prompt_plan_filters_small_unknown_regions_by_quarter_cell_area(self):
        depth = np.ones((20, 80), dtype=np.float32)
        depth[:, :] = 0.0
        depth[1:3, 1:4] = 1.0  # area 6 -> should be filtered when cell_size = 80/16 = 5, quarter-cell area = 6.25
        depth[10:12, 10:14] = 1.0  # area 8 -> should survive

        plan = build_selective_prompt_plan(
            points_3d=torch.zeros((0, 3), dtype=torch.float32),
            points_ins_ids=torch.zeros((0,), dtype=torch.int64),
            depth=depth,
            intrinsics=torch.eye(3, dtype=torch.float32),
            c2w=torch.eye(4, dtype=torch.float32),
            image_shape=(20, 80),
            unknown_grid_cells_per_width=16,
        )

        self.assertEqual(len(plan["unknown_regions"]), 1)
        self.assertEqual(int(plan["unknown_regions"][0]["area"]), 8)

    def test_sampling_helpers_use_mask_boundary_distance_for_border_touching_mask(self):
        mask = np.zeros((9, 9), dtype=bool)
        mask[0:7, 0:7] = True

        component_points = sample_mask_component_points(mask)

        self.assertEqual(component_points.shape[0], 1)
        center = component_points[0]
        self.assertTrue(mask[int(center[1]), int(center[0])])
        self.assertEqual(tuple(center.astype(int)), (3, 3))

    def test_sample_mask_component_points_returns_one_point_per_component(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[1:4, 1:4] = True
        mask[6:9, 6:9] = True

        points = sample_mask_component_points(mask)

        self.assertEqual(points.shape, (2, 2))
        self.assertTrue(mask[int(points[0, 1]), int(points[0, 0])])
        self.assertTrue(mask[int(points[1, 1]), int(points[1, 0])])

    def test_sample_mask_component_points_skips_tiny_component_when_object_has_multiple_components(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:10, 2:10] = True
        mask[15:17, 15:17] = True

        points = sample_mask_component_points(mask)

        self.assertEqual(points.shape, (1, 2))
        self.assertTrue(mask[int(points[0, 1]), int(points[0, 0])])
        self.assertLess(int(points[0, 0]), 12)

    def test_sample_unknown_region_points_uses_bbox_grid_cell_centers_inside_component(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[1:5, 1:5] = True
        mask[5:9, 1:9] = True

        sampled = sample_unknown_region_points(mask, cell_size_px=4.0)

        sampled_int = {tuple(point.astype(int)) for point in sampled}
        self.assertEqual(sampled_int, {(3, 3), (3, 7), (7, 7)})

    def test_save_selective_prompt_debug_views_writes_summary_without_object_mask_files(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        known_mask = np.zeros((8, 8), dtype=bool)
        known_mask[1:3, 1:3] = True
        seen_unknown_mask = np.zeros((8, 8), dtype=bool)
        seen_unknown_mask[4:6, 1:3] = True
        prompt_plan = {
            "known_masks": {3: known_mask},
            "known_union_mask": known_mask.copy(),
            "known_points": {3: np.array([[1.0, 1.0]], dtype=np.float32)},
            "seen_unknown_mask": seen_unknown_mask,
            "seen_unknown_points": np.array([[1.0, 4.0]], dtype=np.float32),
            "seen_unknown_components": [
                {
                    "component_id": 1,
                    "mask": seen_unknown_mask.copy(),
                    "area": int(seen_unknown_mask.sum()),
                    "skipped": True,
                    "skip_reason": "small_area",
                    "n_points": 0,
                    "margin_px": 0,
                    "points": np.zeros((0, 2), dtype=np.float32),
                }
            ],
            "brand_new_unknown_mask": np.logical_not(np.logical_or(known_mask, seen_unknown_mask)),
            "unknown_mask": np.logical_not(known_mask),
            "unknown_regions": [
                {
                    "region_id": 1,
                    "area": int((~np.logical_or(known_mask, seen_unknown_mask)).sum()),
                    "mask": np.logical_not(np.logical_or(known_mask, seen_unknown_mask)),
                    "points": np.array([[5.0, 5.0]], dtype=np.float32),
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_paths = save_selective_prompt_debug_views(
                Path(tmpdir),
                12,
                image,
                prompt_plan,
                sam_seg_map=np.array(
                    [
                        [-1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, -1, -1, -1, -1, -1],
                        [-1, 0, 0, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 1, 1, -1, -1, -1, -1, -1],
                        [-1, 1, 1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1],
                    ],
                    dtype=np.int32,
                ),
                projected_object_ids_view=np.full((8, 8, 3), 17, dtype=np.uint8),
            )

            saved_names = {path.name for path in saved_paths}
            self.assertIn("0012_summary.png", saved_names)
            self.assertEqual(len(saved_names), 1)
            self.assertNotIn("0012_known_points.png", saved_names)
            self.assertNotIn("0012_seen_unknown_points.png", saved_names)
            self.assertNotIn("0012_unknown_points.png", saved_names)
            self.assertNotIn("0012_brand_new_unknown_component_0001_cells.png", saved_names)
            self.assertNotIn("0012_known_dist_object_0003.png", saved_names)
            self.assertNotIn("0012_object_0003_known_mask.png", saved_names)
            self.assertNotIn("0012_known_union.png", saved_names)
            self.assertNotIn("0012_unknown_region.png", saved_names)
            self.assertNotIn("0012_seen_unknown_region.png", saved_names)
            self.assertNotIn("0012_brand_new_unknown_region.png", saved_names)
            self.assertNotIn("0012_known_objects_overlay.png", saved_names)
            self.assertNotIn("0012_seen_unknown_component_0001_skipped.png", saved_names)
            self.assertNotIn("0012_projected_object_ids.png", saved_names)
            self.assertNotIn("0012_all_points.png", saved_names)
            self.assertNotIn("0012_sam_surviving_points.png", saved_names)
            self.assertNotIn("0012_sam_result.png", saved_names)

            summary_path = next(path for path in saved_paths if path.name == "0012_summary.png")
            import cv2

            summary = cv2.cvtColor(cv2.imread(str(summary_path)), cv2.COLOR_BGR2RGB)
            self.assertEqual(summary.shape[:2], (16, 16))

    def test_collect_all_prompt_points_merges_and_deduplicates_all_prompt_sources(self):
        prompt_plan = {
            "known_points": {
                1: np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
                2: np.array([[2.0, 2.0]], dtype=np.float32),
            },
            "seen_unknown_points": np.array([[3.0, 3.0]], dtype=np.float32),
            "unknown_regions": [
                {"points": np.array([[4.0, 4.0], [1.0, 1.0]], dtype=np.float32)},
            ],
        }

        points = collect_all_prompt_points(prompt_plan)

        point_set = {tuple(point.astype(int)) for point in points}
        self.assertEqual(point_set, {(1, 1), (2, 2), (3, 3), (4, 4)})

    def test_build_projected_bbox_prompts_groups_points_by_instance(self):
        points_3d = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [5.0, 5.0, 1.0],
                [6.0, 5.0, 1.0],
                [9.0, 9.0, 1.0],
            ],
            dtype=torch.float32,
        )
        points_ins_ids = torch.tensor([3, 3, 8, 8, -1], dtype=torch.int64)
        intrinsics = torch.eye(3, dtype=torch.float32)
        w2c = torch.eye(4, dtype=torch.float32)

        prompts = build_projected_bbox_prompts(
            points_3d,
            points_ins_ids,
            intrinsics,
            w2c,
            image_shape=(20, 20),
            rgb_depth_ratio=(2.0, 2.0, 1),
            min_points=2,
            padding=1,
        )

        self.assertEqual([prompt["ins_id"] for prompt in prompts], [3, 8])
        np.testing.assert_array_equal(prompts[0]["box"], np.array([3.0, 3.0, 7.0, 5.0], dtype=np.float32))
        np.testing.assert_array_equal(prompts[1]["box"], np.array([11.0, 11.0, 15.0, 13.0], dtype=np.float32))

    def test_build_projected_bbox_prompts_ignores_zero_label_points(self):
        points_3d = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [5.0, 5.0, 1.0],
                [6.0, 5.0, 1.0],
            ],
            dtype=torch.float32,
        )
        points_ins_ids = torch.tensor([0, 0, 2, 2], dtype=torch.int64)
        intrinsics = torch.eye(3, dtype=torch.float32)
        w2c = torch.eye(4, dtype=torch.float32)

        prompts = build_projected_bbox_prompts(
            points_3d,
            points_ins_ids,
            intrinsics,
            w2c,
            image_shape=(20, 20),
            min_points=2,
            padding=0,
        )

        self.assertEqual([prompt["ins_id"] for prompt in prompts], [2])

    def test_filter_point_masks_by_coverage_removes_seed_points_inside_existing_segments(self):
        covered_mask = np.array(
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ],
            dtype=bool,
        )
        point_masks = [
            {
                "segmentation": np.array(
                    [
                        [True, True, False, False],
                        [True, True, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ],
                    dtype=bool,
                ),
                "point_coords": [[0.0, 0.0]],
                "stability_score": 0.9,
                "predicted_iou": 0.9,
            },
            {
                "segmentation": np.array(
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, True, True],
                        [False, False, True, True],
                    ],
                    dtype=bool,
                ),
                "point_coords": [[2.0, 2.0]],
                "stability_score": 0.8,
                "predicted_iou": 0.8,
            },
            {
                "segmentation": np.array(
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, True, True],
                        [False, False, True, True],
                    ],
                    dtype=bool,
                ),
                "point_coords": [[3.0, 3.0]],
                "stability_score": 0.7,
                "predicted_iou": 0.7,
            },
        ]

        filtered_masks, final_coverage = filter_point_masks_by_coverage(point_masks, covered_mask)

        self.assertEqual(len(filtered_masks), 1)
        self.assertEqual(filtered_masks[0]["point_coords"], [[2.0, 2.0]])
        self.assertTrue(final_coverage[2, 2])
        self.assertTrue(final_coverage[3, 3])

    def test_filter_point_masks_by_coverage_ignores_small_uncovered_components(self):
        covered_mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, True, True],
                [True, False, False, True, False],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ],
            dtype=bool,
        )
        point_masks = [
            {
                "segmentation": np.zeros((5, 5), dtype=bool),
                "point_coords": [[1.0, 1.0]],  # large component (area 4)
                "stability_score": 0.8,
                "predicted_iou": 0.8,
            },
            {
                "segmentation": np.zeros((5, 5), dtype=bool),
                "point_coords": [[4.0, 2.0]],  # tiny component (area 1)
                "stability_score": 0.8,
                "predicted_iou": 0.8,
            },
        ]

        filtered_masks, _ = filter_point_masks_by_coverage(
            point_masks,
            covered_mask,
            min_component_area=2,
        )

        self.assertEqual(len(filtered_masks), 1)
        self.assertEqual(filtered_masks[0]["point_coords"], [[1.0, 1.0]])

    def test_filter_point_grid_for_sam_filters_covered_small_and_near_covered(self):
        covered_mask = np.array(
            [
                [True, True, False, False, False],
                [True, True, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, True],
                [False, False, False, False, True],
            ],
            dtype=bool,
        )
        grid_norm = np.array(
            [
                [0.1, 0.1],  # covered
                [0.5, 0.5],  # valid uncovered
                [0.8, 0.8],  # near covered region on bottom-right
            ],
            dtype=np.float32,
        )

        filtered = filter_point_grid_for_sam(
            grid_norm,
            image_shape=(5, 5),
            covered_mask=covered_mask,
            min_component_area=2,
            min_distance_to_covered=1.1,
        )

        self.assertEqual(filtered.shape[0], 1)
        np.testing.assert_allclose(filtered[0], np.array([0.5, 0.5], dtype=np.float32))

    def test_render_debug_views_draws_requested_debug_images(self):
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        views = render_debug_views(
            image=image,
            bbox_prompts=[{"ins_id": 1, "box": np.array([1.0, 1.0, 3.0, 3.0], dtype=np.float32)}],
            point_masks_raw=[
                {
                    "segmentation": np.zeros((5, 5), dtype=bool),
                    "point_coords": [[0.0, 4.0]],
                }
            ],
            point_masks_selected=[
                {
                    "segmentation": np.zeros((5, 5), dtype=bool),
                    "point_coords": [[4.0, 0.0]],
                }
            ],
            point_masks_premerge=[
                {
                    "segmentation": np.array(
                        [
                            [False, False, False, False, False],
                            [False, False, False, False, False],
                            [False, False, True, False, False],
                            [False, False, False, False, False],
                            [False, False, False, False, False],
                        ],
                        dtype=bool,
                    ),
                    "point_coords": [[2.0, 2.0]],
                }
            ],
            point_masks_used=[
                {
                    "segmentation": np.array(
                        [
                            [False, False, False, False, False],
                            [False, False, False, False, False],
                            [False, False, True, True, False],
                            [False, False, True, True, False],
                            [False, False, False, False, False],
                        ],
                        dtype=bool,
                    ),
                    "point_coords": [[2.0, 2.0]],
                }
            ],
            bbox_masks_premerge=[
                {
                    "segmentation": np.array(
                        [
                            [True, False, False, False, False],
                            [True, False, False, False, False],
                            [False, False, False, False, False],
                            [False, False, False, False, False],
                            [False, False, False, False, False],
                        ],
                        dtype=bool,
                    )
                }
            ],
            bbox_masks=[
                {
                    "segmentation": np.array(
                        [
                            [True, True, False, False, False],
                            [True, True, False, False, False],
                            [False, False, False, False, False],
                            [False, False, False, False, False],
                            [False, False, False, False, False],
                        ],
                        dtype=bool,
                    )
                }
            ],
            final_seg_map=np.array(
                [
                    [0, 0, -1, -1, -1],
                    [0, 0, -1, -1, -1],
                    [-1, -1, 1, 1, -1],
                    [-1, -1, 1, 1, -1],
                    [-1, -1, -1, -1, -1],
                ],
                dtype=np.int32,
            ),
            uncovered_mask=np.array(
                [
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, True, True, False],
                    [False, False, True, True, False],
                    [False, False, False, False, False],
                ],
                dtype=bool,
            ),
            valid_uncovered_mask_=np.array(
                [
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, True, True, False],
                    [False, False, True, True, False],
                    [False, False, False, False, False],
                ],
                dtype=bool,
            ),
            initial_point_grid=np.array(
                [
                    [1.0, 1.0],
                    [3.0, 3.0],
                ],
                dtype=np.float32,
            ),
            sam_input_point_grid=np.array(
                [
                    [2.0, 2.0],
                ],
                dtype=np.float32,
            ),
        )

        self.assertEqual(
            set(views.keys()),
            {
                "input",
                "bbox_prompts",
                "point_grid_initial",
                "point_grid_sam_input",
                "point_prompts_raw",
                "point_prompts_selected",
                "point_prompts_used",
                "point_result_premerge",
                "point_result",
                "bbox_result",
                "final_result",
                "uncovered",
                "valid_uncovered",
            },
        )
        self.assertTrue(np.any(views["bbox_prompts"][1, 1] != 0))
        self.assertTrue(np.any(views["point_grid_initial"][1, 1] != 0))
        self.assertTrue(np.any(views["point_grid_sam_input"][2, 2] != 0))
        self.assertTrue(np.any(views["point_prompts_raw"][4, 0] != 0))
        self.assertTrue(np.any(views["point_prompts_selected"][0, 4] != 0))
        self.assertTrue(np.any(views["point_prompts_used"][2, 2] != 0))
        self.assertTrue(np.any(views["point_result_premerge"][2, 2] != 0))
        self.assertTrue(np.any(views["point_result"][2, 2] != 0))
        self.assertTrue(np.any(views["bbox_result"][0, 0] != 0))
        self.assertTrue(np.any(views["final_result"][2, 2] != 0))
        self.assertTrue(np.any(views["uncovered"][2, 2] != 0))
        self.assertTrue(np.any(views["valid_uncovered"][2, 2] != 0))

    def test_merge_bbox_point_masks_merges_by_small_mask_overlap_threshold(self):
        bbox_masks = [
            {
                "segmentation": np.array(
                    [
                        [True, True, False, False],
                        [True, True, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ],
                    dtype=bool,
                ),
                "predicted_iou": 0.5,
                "stability_score": 0.5,
            }
        ]
        point_masks = [
            {
                "segmentation": np.array(
                    [
                        [True, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ],
                    dtype=bool,
                ),
                "predicted_iou": 0.9,
                "stability_score": 0.8,
            },
            {
                "segmentation": np.array(
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, True, True],
                        [False, False, True, True],
                    ],
                    dtype=bool,
                ),
                "predicted_iou": 0.7,
                "stability_score": 0.7,
            },
        ]

        merged_bbox, remaining_points = merge_bbox_point_masks(bbox_masks, point_masks, overlap_th=0.7)

        self.assertEqual(len(merged_bbox), 1)
        self.assertEqual(len(remaining_points), 1)
        self.assertAlmostEqual(float(merged_bbox[0]["predicted_iou"]), 0.9)
        self.assertAlmostEqual(float(merged_bbox[0]["stability_score"]), 0.8)
        self.assertTrue(remaining_points[0]["segmentation"][2, 2])


if __name__ == "__main__":
    unittest.main()
