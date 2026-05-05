import unittest
import sys
import types

import torch

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
sys.modules.setdefault("open_clip", types.ModuleType("open_clip"))
imageio_module = sys.modules.setdefault("imageio", types.ModuleType("imageio"))
imageio_v2_module = sys.modules.setdefault("imageio.v2", types.ModuleType("imageio.v2"))
setattr(imageio_module, "v2", imageio_v2_module)
wandb_module = sys.modules.setdefault("wandb", types.ModuleType("wandb"))
setattr(wandb_module, "log", lambda *args, **kwargs: None)
psutil_module = sys.modules.setdefault("psutil", types.ModuleType("psutil"))
setattr(psutil_module, "Process", lambda *args, **kwargs: None)
setattr(psutil_module, "NoSuchProcess", Exception)
setattr(psutil_module, "Error", Exception)
core_module = sys.modules.setdefault("core", types.ModuleType("core"))
vision_encoder_module = sys.modules.setdefault("core.vision_encoder", types.ModuleType("core.vision_encoder"))
sys.modules.setdefault("core.vision_encoder.pe", types.ModuleType("core.vision_encoder.pe"))
sys.modules.setdefault("core.vision_encoder.transforms", types.ModuleType("core.vision_encoder.transforms"))
setattr(core_module, "vision_encoder", vision_encoder_module)

from ovo.entities.ovo import OVO


class OVOInstanceIdSemanticsTest(unittest.TestCase):
    def test_mark_projected_points_as_seen_promotes_minus_one_to_zero_only_for_matches(self):
        ovo = OVO.__new__(OVO)
        points_ins_ids = torch.tensor([-1, -1, 0, 2], dtype=torch.int64)
        matched_points_idxs = torch.tensor([0, 2, 3], dtype=torch.int64)

        updated = ovo._mark_projected_points_as_seen(points_ins_ids.clone(), matched_points_idxs)

        self.assertEqual(updated.tolist(), [0, -1, 0, 2])

    def test_track_objects_creates_first_object_with_id_one(self):
        ovo = OVO.__new__(OVO)
        ovo.objects = {}
        ovo.next_ins_id = 1

        points_ids = torch.tensor([10, 11, 12], dtype=torch.int64)
        points_ins_ids = torch.tensor([0, 0, 0], dtype=torch.int64)
        matched_points_idxs = torch.tensor([0, 1, 2], dtype=torch.int64)
        matched_seg_idxs = torch.tensor([0, 0, 0], dtype=torch.int64)
        seg_map = torch.zeros((2, 2), dtype=torch.int64)

        updated_ids, matched_info = ovo._track_objects(
            points_ids,
            points_ins_ids,
            matched_points_idxs,
            matched_seg_idxs,
            seg_map,
            track_th=1,
            kf_id=7,
        )

        self.assertEqual(updated_ids.tolist(), [1, 1, 1])
        self.assertEqual(sorted(matched_info.keys()), [1])
        self.assertEqual(ovo.next_ins_id, 2)
        self.assertIn(1, ovo.objects)
