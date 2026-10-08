"""Temporal tracking: players (ByteTrack), puck, goals and rink lines.

All trackers are camera-motion compensated: before associating new detections, the
previous state is warped by the frame-to-frame camera homography, so a fast broadcast
pan doesn't look like every object suddenly jumped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils import matching
from ultralytics.utils import IterableSimpleNamespace

from .camera import warp_points
from .lines import RED_THRESHOLD, RinkLine, drop_crossing, redness


# --------------------------------------------------------------------------- players
class _Dets:
    """Minimal Results-like container accepted by ultralytics' trackers."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, np.float32)
        self.cls = np.asarray(cls, np.float32)

    @property
    def xywh(self) -> np.ndarray:
        b = self.xyxy
        return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_Dets":
        return _Dets(self.xyxy[idx], self.conf[idx], self.cls[idx])


class _ExternalGMC:
    """Feeds our own camera-motion estimate into the tracker's GMC hook."""
    method = "external"

    def __init__(self):
        self.warp = np.eye(2, 3)

    def apply(self, img, dets=None):
        return self.warp

    def reset_params(self):
        self.warp = np.eye(2, 3)


def buffered_iou(a: np.ndarray, b: np.ndarray, buf: float) -> np.ndarray:
    """IoU matrix after growing every box by `buf` times its size on each side (C-BIoU).
    Lets a player still match their track when updates are far enough apart that they've
    moved more than their own width."""
    def grow(x):
        w, h = (x[:, 2] - x[:, 0])[:, None], (x[:, 3] - x[:, 1])[:, None]
        return x + np.hstack([-w, -h, w, h]) * buf
    a, b = grow(a), grow(b)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


class _KeyframeBOTSORT(BOTSORT):
    """BoT-SORT for sparse (keyframe) updates:
      * first-stage matching uses buffered IoU (see buffered_iou);
      * when `flow` is set (camera.Displacement since the previous update), every track is moved
        by the optical flow measured inside its own box instead of by camera motion alone."""

    def __init__(self, args, buffer: float):
        super().__init__(args)
        self.buffer = buffer
        self.flow = None

    def _pre_first_associate(self, strack_pool, unconfirmed, img, results_high) -> None:
        if self.flow is None:
            return super()._pre_first_associate(strack_pool, unconfirmed, img, results_high)
        # pool tracks were just Kalman-predicted (position += velocity): undo that and use the
        # measured motion as both the new position and the velocity estimate
        for t in strack_pool:
            last = t.mean[:2] - t.mean[4:6]
            d = self.flow(np.r_[last - t.mean[2:4] / 2, last + t.mean[2:4] / 2])
            t.mean[:2] = last + d
            t.mean[4:6] = d
        for t in unconfirmed:          # not predicted by the tracker: just move them
            t.mean[:2] = t.mean[:2] + self.flow(t.xyxy)

    def get_dists(self, tracks, detections) -> np.ndarray:
        if self.buffer <= 0 or not tracks or not detections:
            return super().get_dists(tracks, detections)
        a = np.array([t.xyxy for t in tracks], np.float32)
        b = np.array([d.xyxy for d in detections], np.float32)
        dists = 1.0 - buffered_iou(a, b, self.buffer)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists


class PlayerTracker:
    CLASS_IDS = {"player": 0, "goalie": 1, "referee": 2}
    CLASS_NAMES = {v: k for k, v in CLASS_IDS.items()}

    def __init__(self, updates_per_s: float = 30.0, match_buffer: float = 0.0):
        """updates_per_s: how often update() is called per second of video (the keyframe rate).
        match_buffer: box growth for matching (buffered_iou); use >0 when updates are sparse."""
        # BoT-SORT without ReID is ByteTrack association with an XYWH Kalman state, which is
        # what the camera-motion warp (multi_gmc) expects; ByteTrack's XYAH state gets
        # corrupted by it whenever the camera zooms or rolls.
        args = IterableSimpleNamespace(
            tracker_type="botsort", track_high_thresh=0.35, track_low_thresh=0.12,
            new_track_thresh=0.45, track_buffer=max(8, int(updates_per_s * 1.5)), match_thresh=0.8, fuse_score=True,
            gmc_method="none", proximity_thresh=0.5, appearance_thresh=0.8, with_reid=False, model="auto")
        self.tracker = _KeyframeBOTSORT(args, match_buffer)
        self.tracker.gmc = _ExternalGMC()
        self.id_offset = 0
        self.max_id = 0

    def reset(self) -> None:
        """Drop all tracks (after a scene cut). The underlying tracker restarts its IDs at 1,
        so offset them to keep IDs unique across the whole video."""
        self.tracker.reset()
        self.id_offset = self.max_id

    def update(self, boxes: np.ndarray, scores: np.ndarray, labels: list[str],
               camera_affine: np.ndarray, frame: np.ndarray, flow=None) -> list[dict]:
        """flow: optional camera.Displacement since the previous update (used instead of
        camera_affine to predict where each player moved)."""
        self.tracker.gmc.warp = camera_affine
        self.tracker.flow = flow
        cls = np.array([self.CLASS_IDS[l] for l in labels], np.float32)
        out = self.tracker.update(_Dets(boxes, scores, cls), frame)
        tracks = []
        for x1, y1, x2, y2, tid, score, c, idx in out:
            tid = int(tid) + self.id_offset
            self.max_id = max(self.max_id, tid)
            tracks.append({"id": tid, "box": np.array([x1, y1, x2, y2], np.float32),
                           "score": float(score), "cls": self.CLASS_NAMES[int(c)], "det_idx": int(idx)})
        return tracks


# --------------------------------------------------------------------------- puck
@dataclass
class PuckState:
    center: np.ndarray
    velocity: np.ndarray
    size: float
    score: float
    hits: int = 1
    misses: int = 0
    history: list = field(default_factory=list)   # recent centers, current-frame coordinates


class PuckTracker:
    """Single-object tracker. The detector's puck output is noisy (stick blades, skates and
    seat numbers look similar), so a puck is only reported once it's been seen consistently,
    and short gaps are bridged with a constant-velocity prediction."""

    def __init__(self, frame_diag: float, max_missing: int = 8, trail: int = 20, gate_frac: float = 0.10):
        self.gate = gate_frac * frame_diag   # how far the puck may move between updates
        self.max_missing = max_missing
        self.trail = trail
        self.state: PuckState | None = None
        self.candidate: PuckState | None = None

    def reset(self) -> None:
        self.state = self.candidate = None

    def _warp(self, st: PuckState | None, H: np.ndarray) -> None:
        if st is None:
            return
        pts = np.vstack([st.center[None], st.center[None] + st.velocity[None]] + [np.array(st.history).reshape(-1, 2)])
        w = warp_points(H, pts)
        st.center, st.velocity = w[0], w[1] - w[0]
        st.history = [p for p in w[2:]]

    def update(self, candidates: list[tuple[np.ndarray, float]], H: np.ndarray) -> dict | None:
        """candidates: list of (box xyxy, score) already filtered to on-ice pucks."""
        self._warp(self.state, H)
        self._warp(self.candidate, H)
        dets = [((b[:2] + b[2:]) / 2, float(max(b[2] - b[0], b[3] - b[1])), s) for b, s in candidates]

        st = self.state
        if st is not None:
            pred = st.center + st.velocity
            best, best_cost = None, None
            for c, size, s in dets:
                dist = float(np.linalg.norm(c - pred))
                if dist > self.gate * (1 + 0.25 * min(st.misses, 4)):
                    continue
                cost = dist / self.gate - s
                if best_cost is None or cost < best_cost:
                    best, best_cost = (c, size, s), cost
            # a much stronger detection elsewhere wins over a weak continuation
            strongest = max(dets, key=lambda d: d[2], default=None)
            if strongest is not None and strongest[2] >= 0.55 and (best is None or best[2] < 0.25):
                best = strongest
            if best is not None:
                c, size, s = best
                st.history.append(st.center.copy())
                st.history = st.history[-self.trail:]
                new_v = c - st.center
                st.velocity = 0.5 * st.velocity + 0.5 * new_v if st.misses == 0 else new_v / (st.misses + 1)
                st.center, st.size, st.score = c, 0.7 * st.size + 0.3 * size, s
                st.hits += 1
                st.misses = 0
            else:
                st.history.append(st.center.copy())
                st.history = st.history[-self.trail:]
                st.center = pred
                st.velocity *= 0.85
                st.misses += 1
                if st.misses > self.max_missing:
                    self.state = None
            if self.state is not None:
                return self._report(self.state)

        # no confirmed puck: look for one
        if not dets:
            if self.candidate is not None:
                self.candidate.misses += 1
                if self.candidate.misses > 2:
                    self.candidate = None
            return None
        c, size, s = max(dets, key=lambda d: d[2])
        cand = self.candidate
        if cand is not None and np.linalg.norm(c - cand.center) < self.gate * (1 + cand.misses):
            cand.history.append(cand.center.copy())
            cand.velocity = (c - cand.center) / (cand.misses + 1)
            cand.center, cand.size, cand.score = c, size, max(cand.score, s)
            cand.hits += 1
            cand.misses = 0
        else:
            cand = self.candidate = PuckState(c, np.zeros(2), size, s)
        if (cand.hits >= 2 and cand.score >= 0.2) or cand.score >= 0.7:
            self.state, self.candidate = cand, None
            return self._report(cand)
        return None

    @staticmethod
    def _report(st: PuckState) -> dict:
        return {"center": st.center.copy(), "size": st.size, "score": st.score,
                "predicted": st.misses > 0, "trail": [p.copy() for p in st.history]}


# --------------------------------------------------------------------------- goals
@dataclass
class _BoxTrack:
    box: np.ndarray
    score: float
    hits: int = 1
    misses: int = 0


class GoalTracker:
    def __init__(self, max_missing: int = 12, max_goals: int = 2):
        self.tracks: list[_BoxTrack] = []
        self.max_missing = max_missing
        self.max_goals = max_goals

    def reset(self) -> None:
        self.tracks = []

    def update(self, boxes: np.ndarray, scores: np.ndarray, H: np.ndarray) -> list[dict]:
        for t in self.tracks:
            corners = warp_points(H, t.box.reshape(2, 2))
            t.box = corners.reshape(4).astype(np.float32)
        used = set()
        for t in self.tracks:
            best, best_iou = None, 0.2
            for i, b in enumerate(boxes):
                if i in used:
                    continue
                iou = _iou(t.box, b)
                if iou > best_iou:
                    best, best_iou = i, iou
            if best is not None:
                used.add(best)
                t.box = 0.6 * boxes[best] + 0.4 * t.box
                t.score = float(scores[best])
                t.hits += 1
                t.misses = 0
            else:
                t.misses += 1
        for i, b in enumerate(boxes):
            if i not in used:
                self.tracks.append(_BoxTrack(np.asarray(b, np.float32), float(scores[i])))
        self.tracks = [t for t in self.tracks if t.misses <= self.max_missing]
        shown = [t for t in self.tracks if t.hits >= 2 or t.score >= 0.6]
        shown = sorted(shown, key=lambda t: (t.misses, -t.score))[: self.max_goals]
        return [{"box": t.box.copy(), "score": t.score, "predicted": t.misses > 0} for t in shown]


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# --------------------------------------------------------------------------- rink lines
@dataclass
class _LineTrack:
    p1: np.ndarray
    p2: np.ndarray
    thickness: float
    red_sum: float = 0.0     # accumulated color evidence (see lines.redness)
    red_n: float = 0.0
    hits: int = 1
    misses: int = 0

    @property
    def kind(self) -> str:
        return "center_line" if self.red_sum / max(self.red_n, 1e-6) > RED_THRESHOLD else "blue_line"

    @property
    def direction(self) -> np.ndarray:
        d = self.p2 - self.p1
        return d / (np.linalg.norm(d) + 1e-9)


class LineTracker:
    def __init__(self, frame_w: int, max_missing: int = 10, min_hits: int = 3):
        self.tracks: list[_LineTrack] = []
        self.max_missing = max_missing
        self.min_hits = min_hits      # detections needed before a line is shown
        self.dist_tol = 0.035 * frame_w

    def reset(self) -> None:
        self.tracks = []

    def update(self, lines: list[RinkLine], H: np.ndarray,
               center_ice_boxes: np.ndarray | None = None) -> list[dict]:
        """center_ice_boxes: detections of the center-ice circle. The center line runs through
        it, which is a stronger cue than paint color when the footage is washed out."""
        for t in self.tracks:
            t.p1, t.p2 = warp_points(H, np.stack([t.p1, t.p2]))
        used = set()
        for t in self.tracks:
            best, best_d = None, None
            for i, ln in enumerate(lines):
                if i in used:
                    continue
                d = _line_distance(t, ln)
                if d is not None and d < self.dist_tol and (best_d is None or d < best_d):
                    best, best_d = i, d
            if best is not None:
                used.add(best)
                ln = lines[best]
                p1, p2 = _orient(ln.p1, ln.p2, t.direction)
                t.p1 = 0.5 * p1 + 0.5 * t.p1
                t.p2 = 0.5 * p2 + 0.5 * t.p2
                self._vote(t, ln)
                t.thickness = 0.7 * t.thickness + 0.3 * ln.thickness
                t.hits += 1
                t.misses = 0
            else:
                t.misses += 1
        for i, ln in enumerate(lines):
            if i not in used:
                t = _LineTrack(ln.p1.copy(), ln.p2.copy(), ln.thickness)
                self._vote(t, ln)
                self.tracks.append(t)
        self.tracks = [t for t in self.tracks if t.misses <= self.max_missing]

        if center_ice_boxes is not None and len(center_ice_boxes):
            for b in center_ice_boxes:
                c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                near = [t for t in self.tracks if t.misses == 0 and _point_line_dist(c, t) < 0.12 * (b[2] - b[0])]
                if len(near) == 1:
                    near[0].red_sum += 3.0
                    near[0].red_n += 1.0

        shown = [t for t in self.tracks if t.hits >= self.min_hits]
        shown = drop_crossing(shown, key=lambda t: (t.hits - 3 * t.misses))
        # at most one center line and two blue lines; keep the best-supported
        shown.sort(key=lambda t: (t.misses, -t.hits))
        out, counts = [], {"center_line": 0, "blue_line": 0}
        limits = {"center_line": 1, "blue_line": 2}
        for t in shown:
            if counts[t.kind] < limits[t.kind]:
                counts[t.kind] += 1
                out.append({"kind": t.kind, "p1": t.p1.copy(), "p2": t.p2.copy(),
                            "thickness": t.thickness, "predicted": t.misses > 0})
        return out

    @staticmethod
    def _vote(t: _LineTrack, ln: RinkLine) -> None:
        t.red_sum += float(np.clip(redness(ln.delta_lab), -4.0, 4.0))
        t.red_n += 1.0


def _point_line_dist(c: np.ndarray, t: _LineTrack) -> float:
    d = t.direction
    return abs(float((c - t.p1) @ np.array([-d[1], d[0]])))


def _orient(p1, p2, ref_dir):
    d = p2 - p1
    return (p1, p2) if float(d @ ref_dir) >= 0 else (p2, p1)


def _line_distance(t: _LineTrack, ln: RinkLine) -> float | None:
    """Distance between a tracked line and a detection, or None if the angles disagree."""
    dt = t.direction
    dl = (ln.p2 - ln.p1) / (np.linalg.norm(ln.p2 - ln.p1) + 1e-9)
    if abs(float(dt @ dl)) < np.cos(np.deg2rad(7)):
        return None
    n = np.array([-dt[1], dt[0]])
    mid = (ln.p1 + ln.p2) / 2
    return abs(float((mid - t.p1) @ n))
