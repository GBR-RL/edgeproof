import pytest

from edgeproof.metrics import recall_at_far, threshold_at_far


def test_threshold_rejects_at_most_the_allowed_share() -> None:
    normals = [i / 100 for i in range(100)]  # 0.00 ... 0.99
    assert threshold_at_far(normals, 0.05) == pytest.approx(0.94)
    assert threshold_at_far(normals, 0.0) == pytest.approx(0.99)
    assert threshold_at_far(normals, 1.0) == 0.0


def test_recall_at_far_counts_defects_above_the_threshold() -> None:
    normals = [0.1] * 95 + [0.9] * 5
    defects = [0.95, 0.85, 0.5, 0.05]
    strict = recall_at_far(defects, normals, far=0.01)
    loose = recall_at_far(defects, normals, far=0.05)
    assert strict["recall"] == 0.25  # threshold 0.9: only 0.95 passes
    assert strict["far"] == 0.0
    assert loose["recall"] == 0.75  # threshold 0.1
    assert loose["far"] == 0.05
