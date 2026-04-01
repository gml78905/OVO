import os
import unittest
from pathlib import Path

from ovo.utils import path_utils


class PathUtilsTest(unittest.TestCase):
    def setUp(self):
        self.original_data_root = os.environ.get("OVO_DATA_ROOT")
        os.environ["OVO_DATA_ROOT"] = "/tmp/ovo-data-root"

    def tearDown(self):
        if self.original_data_root is None:
            os.environ.pop("OVO_DATA_ROOT", None)
        else:
            os.environ["OVO_DATA_ROOT"] = self.original_data_root

    def test_resolve_data_path_uses_runtime_root(self):
        resolved = path_utils.resolve_data_path("data/input/Datasets/Replica/office0")
        self.assertEqual(
            resolved,
            Path("/tmp/ovo-data-root/input/Datasets/Replica/office0"),
        )

    def test_remap_data_paths_only_rewrites_data_prefixed_strings(self):
        config = {
            "slam": {"config_path": "data/working/configs/slam"},
            "semantic": {
                "sam": {"sam_ckpt_path": "data/input/sam_ckpts/"},
                "name": "leave-me-alone",
            },
            "items": ["./data/output/Replica", "relative/path"],
        }

        remapped = path_utils.remap_data_paths(config)

        self.assertEqual(
            remapped["slam"]["config_path"],
            "/tmp/ovo-data-root/working/configs/slam",
        )
        self.assertEqual(
            remapped["semantic"]["sam"]["sam_ckpt_path"],
            "/tmp/ovo-data-root/input/sam_ckpts",
        )
        self.assertEqual(
            remapped["items"][0],
            "/tmp/ovo-data-root/output/Replica",
        )
        self.assertEqual(remapped["semantic"]["name"], "leave-me-alone")
        self.assertEqual(remapped["items"][1], "relative/path")
