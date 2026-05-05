import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ovo.utils.live_instance_vis import build_live_instance_debug_image, save_live_instance_debug_image


class LiveInstanceVisualizationTest(unittest.TestCase):
    def test_build_live_instance_debug_image_renders_projected_points(self):
        image = np.zeros((6, 6, 3), dtype=np.uint8)
        image[..., 1] = 25
        depth = np.ones((6, 6), dtype=np.float32)
        points_3d = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [4.0, 2.0, 1.0],
                [2.0, 4.0, 1.0],
            ],
            dtype=torch.float32,
        )
        point_instance_ids = torch.tensor([3, 7, -1], dtype=torch.int32)
        intrinsics = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        c2w = torch.eye(4, dtype=torch.float32)
        seg_map = torch.tensor(
            [
                [0, 0, -1, -1, -1, -1],
                [0, 0, -1, -1, -1, -1],
                [-1, -1, 1, 1, -1, -1],
                [-1, -1, 1, 1, -1, -1],
                [-1, -1, -1, -1, 2, 2],
                [-1, -1, -1, -1, 2, 2],
            ],
            dtype=torch.int64,
        )

        debug_image = build_live_instance_debug_image(
            image=image,
            depth=depth,
            points_3d=points_3d,
            point_instance_ids=point_instance_ids,
            intrinsics=intrinsics,
            c2w=c2w,
            seg_map=seg_map,
            match_distance_th=0.05,
        )

        self.assertEqual(debug_image.shape, (12, 12, 3))
        self.assertEqual(debug_image.dtype, np.uint8)
        live_panel = debug_image[6:, 6:, :]
        self.assertGreater(np.count_nonzero(live_panel[..., 0]), 0)
        self.assertGreater(np.count_nonzero(live_panel[..., 1]), 0)

    def test_save_live_instance_debug_image_writes_png(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        depth = np.ones((4, 4), dtype=np.float32)
        points_3d = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        point_instance_ids = torch.tensor([5], dtype=torch.int32)
        intrinsics = torch.eye(3, dtype=torch.float32)
        c2w = torch.eye(4, dtype=torch.float32)
        seg_map = torch.zeros((4, 4), dtype=torch.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = save_live_instance_debug_image(
                output_dir=Path(tmpdir),
                frame_id=12,
                image=image,
                depth=depth,
                points_3d=points_3d,
                point_instance_ids=point_instance_ids,
                intrinsics=intrinsics,
                c2w=c2w,
                seg_map=seg_map,
                match_distance_th=0.05,
            )

            self.assertEqual(output_path.name, "0012_live_instance_debug.png")
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
