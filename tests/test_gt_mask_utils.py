import unittest

import numpy as np

from ovo.utils.gt_mask_utils import (
    backproject_depth_to_world,
    build_vertex_object_ids,
    object_id_map_to_masks,
)


class GTMaskUtilsTest(unittest.TestCase):
    def test_build_vertex_object_ids_uses_face_majority_vote(self):
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [1, 2, 4],
            ],
            dtype=np.int64,
        )
        face_object_ids = np.array([10, 10, 7], dtype=np.int32)

        vertex_object_ids = build_vertex_object_ids(5, faces, face_object_ids)

        np.testing.assert_array_equal(
            vertex_object_ids,
            np.array([10, 7, 10, 10, 7], dtype=np.int32),
        )

    def test_object_id_map_to_masks_compacts_ids_and_filters_small_regions(self):
        object_id_map = np.array(
            [
                [10, 10, -1, 7],
                [10, -1, 7, 7],
            ],
            dtype=np.int32,
        )

        seg_map, binary_maps, object_ids = object_id_map_to_masks(object_id_map, min_area=2)

        np.testing.assert_array_equal(
            seg_map,
            np.array(
                [
                    [1, 1, -1, 0],
                    [1, -1, 0, 0],
                ],
                dtype=np.int32,
            ),
        )
        self.assertEqual(binary_maps.shape, (2, 2, 4))
        np.testing.assert_array_equal(object_ids, np.array([7, 10], dtype=np.int32))

    def test_backproject_depth_to_world_respects_intrinsics_and_pose(self):
        depth = np.array([[2.0]], dtype=np.float32)
        intrinsics = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        points = backproject_depth_to_world(depth, intrinsics, c2w)

        np.testing.assert_allclose(points, np.array([[1.0, 2.0, 5.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
