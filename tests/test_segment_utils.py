import unittest

import numpy as np

from ovo.utils.segment_utils import extract_unique_mask_points, generate_masks_from_points


class _FakeMaskGenerator:
    def __init__(self):
        self.point_grids = [np.array([[0.25, 0.25]], dtype=np.float32)]
        self.calls = []

    def generate(self, image):
        self.calls.append(
            {
                "image_shape": image.shape[:2],
                "point_grids": [grid.copy() for grid in self.point_grids],
            }
        )
        return [{"segmentation": np.ones(image.shape[:2], dtype=bool)}]


class SegmentUtilsTest(unittest.TestCase):
    def test_generate_masks_from_points_uses_normalized_custom_grid_and_restores_original_grid(self):
        generator = _FakeMaskGenerator()
        image = np.zeros((20, 40, 3), dtype=np.uint8)
        points = np.array([[10.0, 5.0], [20.0, 10.0]], dtype=np.float32)

        masks = generate_masks_from_points(generator, image, points)

        self.assertEqual(len(masks), 1)
        self.assertEqual(len(generator.calls), 1)
        np.testing.assert_allclose(
            generator.calls[0]["point_grids"][0],
            np.array([[0.25, 0.25], [0.5, 0.5]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            generator.point_grids[0],
            np.array([[0.25, 0.25]], dtype=np.float32),
        )

    def test_generate_masks_from_points_returns_empty_for_no_points(self):
        generator = _FakeMaskGenerator()
        image = np.zeros((20, 40, 3), dtype=np.uint8)

        masks = generate_masks_from_points(generator, image, np.zeros((0, 2), dtype=np.float32))

        self.assertEqual(masks, [])
        self.assertEqual(generator.calls, [])

    def test_extract_unique_mask_points_deduplicates_by_point_coords(self):
        masks = [
            {"point_coords": [[1.0, 2.0]]},
            {"point_coords": [[1.0, 2.0]]},
            {"point_coords": [[3.0, 4.0]]},
        ]

        points = extract_unique_mask_points(masks)

        point_set = {tuple(point.astype(int)) for point in points}
        self.assertEqual(point_set, {(1, 2), (3, 4)})
