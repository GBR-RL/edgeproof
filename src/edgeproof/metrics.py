"""The line's view of accuracy: of the defective parts, how many are caught when good parts are
rejected at a fixed, tolerable rate.

mAP rewards outlining every defect region; a line mostly needs the part sorted out. Each image is
scored by its most confident detection, the threshold is set on the defect-free test images so
that at most ``far`` of them would be rejected, and recall is the share of defective images
scored above that threshold.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def threshold_at_far(normal_scores: Sequence[float], far: float) -> float:
    """The lowest threshold that rejects at most ``far`` of the defect-free images.

    An image is rejected when its score is strictly above the threshold.
    """
    scores = np.sort(np.asarray(normal_scores, dtype=float))[::-1]
    allowed = int(np.floor(far * len(scores)))  # images that may be rejected
    if allowed >= len(scores):
        return 0.0
    return float(scores[allowed])


def recall_at_far(
    defect_scores: Sequence[float], normal_scores: Sequence[float], far: float = 0.01
) -> dict[str, float]:
    threshold = threshold_at_far(normal_scores, far)
    defects = np.asarray(defect_scores, dtype=float)
    normals = np.asarray(normal_scores, dtype=float)
    return {
        "far_target": far,
        "threshold": round(threshold, 5),
        "recall": round(float((defects > threshold).mean()), 5),
        "far": round(float((normals > threshold).mean()), 5),
    }
