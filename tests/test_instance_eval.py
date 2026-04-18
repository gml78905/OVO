import unittest

import numpy as np

from ovo.utils.instance_eval import (
    compute_ap,
    evaluate_ap50,
    evaluate_instance_miou,
    InstanceGroundTruth,
    InstancePrediction,
)


class InstanceEvalTest(unittest.TestCase):
    def test_compute_ap_perfect(self):
        ap = compute_ap(np.array([1.0]), np.array([1.0]))
        self.assertAlmostEqual(ap, 1.0)

    def test_evaluate_ap50_single_match(self):
        pred = InstancePrediction(scene="office0", class_id=3, score=0.9, face_mask=np.array([True, False, False]))
        gt = InstanceGroundTruth(scene="office0", class_id=3, object_id=10, face_mask=np.array([True, False, False]))

        per_class_ap, map50 = evaluate_ap50([pred], [gt], iou_threshold=0.5)

        self.assertAlmostEqual(per_class_ap[3], 1.0)
        self.assertAlmostEqual(map50, 1.0)

    def test_evaluate_instance_miou_single_match(self):
        pred = InstancePrediction(scene="office0", class_id=3, score=0.9, face_mask=np.array([True, False, False]))
        gt = InstanceGroundTruth(scene="office0", class_id=3, object_id=10, face_mask=np.array([True, False, False]))

        miou = evaluate_instance_miou([pred], [gt])

        self.assertAlmostEqual(miou, 1.0)


if __name__ == "__main__":
    unittest.main()
