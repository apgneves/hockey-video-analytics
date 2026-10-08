"""Frame normalization against the ice.

The ice is white in reality, but arena footage is often tinted (e.g. strongly blue) and
under-exposed. Scaling each channel so the ice becomes a fixed bright neutral makes the
detector (trained on well-lit broadcasts) and jersey colors behave consistently across
arenas.
"""
from __future__ import annotations

import cv2
import numpy as np

ICE_TARGET = 235.0


def ice_gains(frame_bgr: np.ndarray, target: float | None = ICE_TARGET) -> np.ndarray:
    """Per-channel gains that map the ice (the dominant bright surface) to a neutral color.
    With target=None only the color cast is removed; otherwise brightness is normalized too."""
    h, w = frame_bgr.shape[:2]
    step = max(1, w // 320)                  # a sparse pixel sample is plenty for statistics
    small = frame_bgr[h // 4::step, ::step].reshape(-1, 3).astype(np.float32)
    lum = small.mean(1)
    bright = small[lum > np.percentile(lum, 70)]
    ref = np.median(bright, 0) if len(bright) else np.array([200.0, 200.0, 200.0])
    goal = ref.mean() if target is None else target
    return np.clip(goal / np.maximum(ref, 1.0), 0.6, 1.8).astype(np.float32)


def ice_fraction(frame_bgr: np.ndarray, gains: np.ndarray) -> float:
    """Share of the lower frame that looks like ice. Game footage (even close-ups) is >25%;
    crowd shots, intermission studio shots and most commercials are near zero."""
    h, w = frame_bgr.shape[:2]
    step = max(1, w // 320)
    s = frame_bgr[h // 4::step, ::step].astype(np.float32) * gains
    lo, hi = s.min(2), s.max(2)
    return float(((lo > 185) & (hi - lo < 35)).mean())


def apply_gains(frame_bgr: np.ndarray, gains: np.ndarray) -> np.ndarray:
    if np.allclose(gains, 1.0, atol=0.03):
        return frame_bgr
    return cv2.transform(frame_bgr, np.diag(gains).astype(np.float32))   # saturating, SIMD
