import tempfile
import unittest
from pathlib import Path

import numpy as np

from ovo.utils.eval_utils import read_label_file


class ReadLabelFileTests(unittest.TestCase):
    def test_reads_utf8_text_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "labels.txt"
            path.write_text("1\n2\n3\n", encoding="utf-8")

            labels = read_label_file(path)

            np.testing.assert_array_equal(labels, np.array([1, 2, 3], dtype=np.int64))

    def test_reads_utf16_text_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "labels.txt"
            path.write_text("4\n5\n6\n", encoding="utf-16")

            labels = read_label_file(path)

            np.testing.assert_array_equal(labels, np.array([4, 5, 6], dtype=np.int64))

    def test_reads_binary_int32_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "labels.bin"
            np.array([7, 8, 9], dtype=np.int32).tofile(path)

            labels = read_label_file(path)

            np.testing.assert_array_equal(labels, np.array([7, 8, 9], dtype=np.int64))
