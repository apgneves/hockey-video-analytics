"""Team assignment from jersey colors.

Each skater's torso is summarized by a few robust Lab color statistics (median
lightness/chroma, fraction of colored, dark and white pixels). Features are standardized
and clustered into two teams with k-means. Callers pass frames normalized against the ice
(color.py) so a blue-tinted or dim arena doesn't make white jerseys look blue or gray.

The model is fit on frames sampled across the whole video, then every track votes over
its lifetime so a single mis-colored frame (occlusion, motion blur) doesn't flip it.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def torso_feature(frame_bgr: np.ndarray, box) -> np.ndarray | None:
    """Lab color statistics of the central jersey region of a person box.

    Returns [median L, median a, median b, 75th pct a, 25th pct L,
             colored fraction, dark fraction, white fraction]."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 16:
        return None
    H, W = frame_bgr.shape[:2]
    fx1, fx2 = max(x1 + int(0.25 * w), 0), min(x2 - int(0.25 * w), W)
    fy1, fy2 = max(y1 + int(0.2 * h), 0), min(y1 + int(0.5 * h), H)
    crop = frame_bgr[fy1:fy2, fx1:fx2]
    if crop.size < 30:
        return None
    lab = cv2.cvtColor(crop.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    chroma = np.hypot(a, b)
    return np.array([np.median(L), np.median(a), np.median(b), np.percentile(a, 75), np.percentile(L, 25),
                     (chroma > 18).mean(), (L < 35).mean(), ((L > 70) & (chroma < 12)).mean()], np.float32)


# Lab hue angle (degrees) upper bounds -> color name, display BGR. Washed-out broadcast reds
# drift toward magenta, so "red" covers a wide band.
_HUES = [(60, "red", (40, 40, 220)), (80, "orange", (0, 130, 255)), (110, "gold", (0, 200, 240)),
         (170, "green", (60, 170, 40)), (220, "teal", (170, 160, 0)), (300, "blue", (220, 110, 20)),
         (325, "purple", (160, 50, 130)), (361, "red", (40, 40, 220))]
_F_L, _F_A, _F_B, _F_A75, _F_L25, _F_COLORED, _F_DARK, _F_WHITE = range(8)


def describe_jersey(f: np.ndarray) -> tuple[str, tuple[int, int, int]]:
    """Human-readable name and a clear display color from a cluster's median torso_feature."""
    if f[_F_WHITE] >= 0.3 and f[_F_WHITE] >= f[_F_COLORED]:
        return "white", (245, 245, 245)
    if f[_F_COLORED] >= 0.25 and np.hypot(f[_F_A], f[_F_B]) > 5:
        hue = float(np.degrees(np.arctan2(f[_F_B], f[_F_A]))) % 360
        for upper, name, bgr in _HUES:
            if hue < upper:
                return name, bgr
    # achromatic jersey: hockey teams wear white or black/dark, rarely true gray
    return ("white", (245, 245, 245)) if f[_F_WHITE] >= f[_F_DARK] else ("black", (40, 40, 40))


def is_colored_jersey(f: np.ndarray | None) -> bool:
    """Referees wear black-and-white stripes; a clearly colored torso is a player."""
    return f is not None and f[_F_COLORED] > 0.28


class TeamClassifier:
    def __init__(self):
        self.kmeans: KMeans | None = None
        self.scaler: StandardScaler | None = None
        self.samples: list[np.ndarray] = []
        self.team_colors: list[tuple[int, int, int]] = []
        self.team_names: list[str] = []
        self.votes: dict[int, Counter] = defaultdict(Counter)

    @property
    def ready(self) -> bool:
        return self.kmeans is not None

    def add_samples(self, frame_bgr: np.ndarray, boxes: np.ndarray) -> None:
        for b in boxes:
            f = torso_feature(frame_bgr, b)
            if f is not None:
                self.samples.append(f)

    def fit(self) -> bool:
        if len(self.samples) < 6:
            return False
        X = np.array(self.samples)
        self.scaler = StandardScaler().fit(X)
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(self.scaler.transform(X))
        meds = [np.median(X[self.kmeans.labels_ == k], axis=0) for k in range(2)]
        described = [describe_jersey(m) for m in meds]
        self.team_names = [d[0] for d in described]
        self.team_colors = [d[1] for d in described]
        if self.team_names[0] == self.team_names[1]:
            # Same hue family (e.g. white-with-red-trim vs red-and-black in a dim arena where
            # nothing reads as pure white): name the pair by contrast instead.
            hi, lo = (0, 1) if meds[0][_F_L] >= meds[1][_F_L] else (1, 0)
            if meds[hi][_F_L] - meds[lo][_F_L] > 15 and meds[hi][_F_COLORED] <= meds[lo][_F_COLORED]:
                self.team_names[hi], self.team_colors[hi] = "white", (245, 245, 245)
            elif meds[hi][_F_L] - meds[lo][_F_L] > 15:
                self.team_names[lo], self.team_colors[lo] = "black", (40, 40, 40)
            else:
                self.team_names = [f"{self.team_names[0]} A", f"{self.team_names[1]} B"]
                self.team_colors[1] = tuple(int(0.6 * v) for v in self.team_colors[1])
        return True

    def predict(self, frame_bgr: np.ndarray, boxes) -> list[int | None]:
        """Team index (0/1) for each box, or None when the jersey isn't visible enough."""
        if self.kmeans is None or len(boxes) == 0:
            return [None] * len(boxes)
        feats = [torso_feature(frame_bgr, b) for b in boxes]
        ok = [i for i, f in enumerate(feats) if f is not None]
        out: list[int | None] = [None] * len(boxes)
        if ok:
            labels = self.kmeans.predict(self.scaler.transform(np.array([feats[i] for i in ok])))
            for i, lab in zip(ok, labels):
                out[i] = int(lab)
        return out

    def vote(self, track_id: int, team: int | None, weight: float = 1.0) -> int | None:
        if team is not None:
            self.votes[track_id][team] += weight
        c = self.votes.get(track_id)
        return c.most_common(1)[0][0] if c else None
