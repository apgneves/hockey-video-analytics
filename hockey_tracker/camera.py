"""Frame-to-frame camera motion (pan / tilt / zoom) and scene-cut detection.

Broadcast cameras rotate about a fixed point, so consecutive frames are related by a
homography regardless of scene depth. We track sparse corners with Lucas-Kanade optical
flow (ignoring players, who move independently) and fit that homography with RANSAC.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_W = 640
DUPLICATE_DIFF = 0.25   # mean absolute gray-level difference below which a frame is a repeat


@dataclass
class Motion:
    H: np.ndarray          # 3x3 homography, previous frame -> current frame (full-res pixels)
    affine: np.ndarray     # 2x3 similarity approximation (for the player tracker)
    scene_cut: bool
    n_points: int
    duplicate: bool = False   # identical to the previous frame (padded / screen-recorded video)


class CameraMotion:
    def __init__(self, max_corners: int = 400):
        self.max_corners = max_corners
        self.prev_gray: np.ndarray | None = None
        self.prev_hist: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_hist = None

    def update(self, frame_bgr: np.ndarray, exclude_boxes: np.ndarray | None = None) -> Motion:
        H0, W0 = frame_bgr.shape[:2]
        s = WORK_W / W0
        small = cv2.resize(frame_bgr, (WORK_W, int(round(H0 * s))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)

        identity = Motion(np.eye(3), np.eye(2, 3), False, 0)
        prev_gray, prev_hist = self.prev_gray, self.prev_hist
        self.prev_gray, self.prev_hist = gray, hist
        if prev_gray is None:
            return identity
        # Many recordings repeat frames (e.g. 15 fps content in a 30 fps file). A repeat differs
        # only by re-encoding noise (<0.1 gray levels on average); real motion is >0.4.
        if cv2.absdiff(prev_gray, gray).mean() < DUPLICATE_DIFF:
            return Motion(np.eye(3), np.eye(2, 3), False, 0, duplicate=True)

        hist_change = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)

        mask = np.full(gray.shape, 255, np.uint8)
        if exclude_boxes is not None:
            for x1, y1, x2, y2 in np.asarray(exclude_boxes, np.float32).reshape(-1, 4) * s:
                mask[max(0, int(y1)):int(y2) + 1, max(0, int(x1)):int(x2) + 1] = 0
        pts = cv2.goodFeaturesToTrack(prev_gray, self.max_corners, 0.01, 8, mask=mask, blockSize=7)
        if pts is None or len(pts) < 12:
            return Motion(np.eye(3), np.eye(2, 3), hist_change > 0.45, 0)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        p0, p1 = pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
        if len(p0) < 12:
            return Motion(np.eye(3), np.eye(2, 3), True, len(p0))

        Hs, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 2.0)
        n_inl = int(inl.sum()) if inl is not None else 0
        # a hard cut: colors change a lot and the flow no longer explains the scene
        cut = hist_change > 0.45 or (hist_change > 0.25 and n_inl < 0.3 * len(p0))
        if Hs is None or cut:
            return Motion(np.eye(3), np.eye(2, 3), cut, n_inl)
        A, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if A is None:
            A = Hs[:2].copy()

        S, Si = np.diag([s, s, 1.0]), np.diag([1 / s, 1 / s, 1.0])
        H_full = Si @ Hs @ S
        A_full = (Si @ np.vstack([A, [0, 0, 1]]) @ S)[:2]
        return Motion(H_full, A_full, False, n_inl)


class Displacement:
    """Dense motion from one keyframe to the next (low resolution). Unlike camera motion it
    includes the players' own skating: broadcast cameras follow the play, so a skater can
    stay put in the image while the ice slides underneath."""

    def __init__(self, field_: np.ndarray, scale: float):
        self.field, self.scale = field_, scale

    def __call__(self, box) -> np.ndarray:
        """Full-resolution (dx, dy) of the jersey region of a box given at the first keyframe."""
        x1, y1, x2, y2 = np.asarray(box, np.float32) * self.scale
        w, h = x2 - x1, y2 - y1
        H, W = self.field.shape[:2]
        ya, yb = int(np.clip(y1 + 0.2 * h, 0, H - 1)), int(np.clip(y1 + 0.8 * h, 0, H - 1)) + 1
        xa, xb = int(np.clip(x1 + 0.25 * w, 0, W - 1)), int(np.clip(x2 - 0.25 * w, 0, W - 1)) + 1
        region = self.field[ya:yb, xa:xb].reshape(-1, 2)
        return np.median(region, 0) / self.scale if len(region) else np.zeros(2, np.float32)


class FlowAccumulator:
    """DIS optical flow on every frame, accumulated since the last keyframe (~7 ms/frame)."""
    WORK_W = 480

    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.prev: np.ndarray | None = None
        self.acc: np.ndarray | None = None
        self.grid: tuple[np.ndarray, np.ndarray] | None = None
        self.scale = 1.0

    def update(self, frame_bgr: np.ndarray, duplicate: bool) -> None:
        H0, W0 = frame_bgr.shape[:2]
        self.scale = self.WORK_W / W0
        gray = cv2.cvtColor(cv2.resize(frame_bgr, (self.WORK_W, int(round(H0 * self.scale))),
                                       interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if self.acc is None or self.acc.shape[:2] != gray.shape:
            self.acc = np.zeros(gray.shape + (2,), np.float32)
            gy, gx = np.mgrid[0:gray.shape[0], 0:gray.shape[1]].astype(np.float32)
            self.grid = (gx, gy)
        if self.prev is not None and not duplicate:
            step = self.dis.calc(self.prev, gray, None)
            gx, gy = self.grid
            # follow each keyframe pixel along its path: sample this step's flow where it is now
            self.acc += cv2.remap(step, gx + self.acc[..., 0], gy + self.acc[..., 1], cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        self.prev = gray

    def take(self) -> Displacement:
        """Motion since the previous take() (i.e. keyframe to keyframe); restarts accumulation."""
        field_ = self.acc if self.acc is not None else np.zeros((1, 1, 2), np.float32)
        self.acc = None if self.acc is None else np.zeros_like(self.acc)
        return Displacement(field_, self.scale)


def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)
