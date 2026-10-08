"""Rink line detection: blue lines and the center red line.

Broadcast footage washes out rink paint, so lines are found as long, straight, *thick*
stripes that are darker than the surrounding ice (blue and red lines are 12" wide; the
faceoff circles, goal lines and crease are 2" wide and are filtered out by thickness).
Each stripe is then classified as blue or red from its color relative to the
neighbouring ice, which is robust to the strong white-balance casts seen in arenas.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_W = 960  # processing width


@dataclass
class RinkLine:
    kind: str                 # "blue_line" | "center_line"
    p1: np.ndarray            # endpoint (x, y), full-resolution pixels
    p2: np.ndarray
    thickness: float          # full-resolution pixels
    delta_lab: tuple[float, float, float]  # stripe color minus ice color (L, a, b)
    support: float            # visible length backed by stripe pixels, work-res pixels
    side_ice: float           # min fraction of ice on the two sides of the stripe

    @property
    def angle(self) -> float:
        d = self.p2 - self.p1
        return float(np.arctan2(d[1], d[0]))

    def as_dict(self) -> dict:
        return {"kind": self.kind,
                "p1": [round(float(v), 1) for v in self.p1],
                "p2": [round(float(v), 1) for v in self.p2],
                "thickness": round(self.thickness, 1)}


def _odd(n: float) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def _disk(n: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))


def _rect(n: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))


def estimate_ice(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (ice mask, ice Lab color). The ice color is estimated per frame so strong
    color casts (e.g. blue-tinted arenas) are handled."""
    h = lab.shape[0]
    lower = lab[h // 5::2, ::2]           # statistics only: a quarter of the pixels is plenty
    L = lower[..., 0]
    bright = L > np.percentile(L, 55)
    a0 = float(np.median(lower[..., 1][bright]))
    b0 = float(np.median(lower[..., 2][bright]))
    near = bright & (np.abs(lower[..., 1] - a0) < 8) & (np.abs(lower[..., 2] - b0) < 8)
    L0 = float(np.percentile(L[near], 60)) if near.any() else float(np.percentile(L, 70))
    ice = ((np.abs(lab[..., 1] - a0) < 10) & (np.abs(lab[..., 2] - b0) < 12)
           & (lab[..., 0] > L0 - 45))
    return ice, np.array([L0, a0, b0], np.float32)


def ice_region(ice: np.ndarray) -> np.ndarray:
    """Solid ice surface: close small gaps (lines, players) and fill enclosed holes."""
    h, w = ice.shape
    m = ice.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _rect(5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _rect(_odd(0.02 * w)))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    big = stats[:, cv2.CC_STAT_AREA] > 0.03 * h * w
    big[0] = False
    keep = big[lbl]
    # fill holes (players / logos fully surrounded by ice)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats((~keep).astype(np.uint8), connectivity=4)
    x, y, ww, hh, area = stats.T
    touches = (x == 0) | (y == 0) | (x + ww >= w) | (y + hh >= h)
    fill = ~touches | (area < 0.01 * h * w)
    fill[0] = False
    return keep | fill[lbl]


@dataclass
class _Piece:
    center: np.ndarray
    direction: np.ndarray
    length: float
    width: float
    pixels: np.ndarray        # (N, 2) float32 x, y


class RinkLineDetector:
    """Finds blue lines and the center line in a single frame."""

    def __init__(self, min_angle_deg: float = 18.0, max_angle_deg: float = 86.5):
        self.min_sin = np.sin(np.deg2rad(min_angle_deg))
        # glass stanchions are exactly vertical; painted lines seen from a raised camera are not
        self.max_sin = np.sin(np.deg2rad(max_angle_deg))
        self.debug: dict = {}

    def detect(self, frame_bgr: np.ndarray, person_boxes: np.ndarray | None = None) -> list[RinkLine]:
        H0, W0 = frame_bgr.shape[:2]
        s = WORK_W / W0
        img = cv2.resize(frame_bgr, (WORK_W, int(round(H0 * s))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        # Lab with L on 0-255 and a*/b* centered on 0. The 8-bit conversion is ~15x faster than
        # float; paint/ice color deltas are trimmed means over hundreds of pixels, which recovers
        # sub-unit precision from the integer values.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 1:] -= 128.0
        ice, ice_lab = estimate_ice(lab)
        region = ice_region(ice)

        people = np.zeros((h, w), bool)
        if person_boxes is not None:
            for x1, y1, x2, y2 in np.asarray(person_boxes, np.float32).reshape(-1, 4) * s:
                bw, bh = x2 - x1, y2 - y1
                people[max(0, int(y1 - 0.05 * bh)):int(y2 + 0.05 * bh) + 1,
                       max(0, int(x1 - 0.1 * bw)):int(x2 + 0.1 * bw) + 1] = True

        L8 = np.clip(lab[..., 0], 0, 255).astype(np.uint8)
        # disk kernel on purpose: a (much faster) square one also fires on the corners of logo
        # lettering, which then lines up into false lines
        blackhat = cv2.morphologyEx(L8, cv2.MORPH_BLACKHAT, _disk(_odd(0.045 * w)))
        stripe = (blackhat > 9) & region & ~people & (lab[..., 0] > ice_lab[0] - 90)
        stripe = stripe.astype(np.uint8) * 255
        # keep only thick paint (blue / center lines), drop 2" lines (circles, goal line)
        thick = cv2.morphologyEx(stripe, cv2.MORPH_OPEN, _disk(_odd(0.006 * w)))
        self.debug = {"region": region, "stripe": stripe > 0, "thick": thick > 0, "scale": s}

        pieces = self._pieces(thick, h)
        lines = []
        for group in _group_collinear(pieces, tol=0.012 * w):
            line = self._fit(group, lab, region, people, w, h)
            if line is not None:
                lines.append(line)
        lines = drop_crossing(lines, key=lambda l: l.support * l.side_ice)
        for line in lines:
            line.p1, line.p2 = line.p1 / s, line.p2 / s
            line.thickness /= s
        return self._classify(lines)

    def _pieces(self, thick: np.ndarray, h: int) -> list[_Piece]:
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(thick, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            # a piece can't be longer than its bounding-box diagonal: skip small ones cheaply
            if area < 40 or np.hypot(bw, bh) < 0.06 * h:
                continue
            ys, xs = np.nonzero(lbl[y:y + bh, x:x + bw] == i)
            pts = np.stack([xs + x, ys + y], 1).astype(np.float32)
            c = pts.mean(0)
            cov = np.cov((pts - c).T)
            evals, evecs = np.linalg.eigh(cov)
            d = evecs[:, 1]
            t = (pts - c) @ d
            length = float(t.max() - t.min())
            width = len(pts) / max(length, 1.0)
            perp = (pts - c) @ np.array([-d[1], d[0]])
            if length < 0.06 * h or length < 4 * width:
                continue
            if abs(d[1]) < self.min_sin:      # too close to horizontal: boards / kick plate
                continue
            if abs(d[1]) > self.max_sin:      # vertical: glass stanchions
                continue
            if perp.std() > 0.45 * width + 1.5:   # curved (circle arcs, logo outlines)
                continue
            out.append(_Piece(c, d, length, width, pts))
        return out

    def _fit(self, group: list[_Piece], lab, region, people, w, h) -> RinkLine | None:
        pts = np.concatenate([p.pixels for p in group])
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        d = np.array([vx, vy], np.float32); n = np.array([-vy, vx], np.float32); c = np.array([x0, y0], np.float32)
        t = (pts - c) @ d
        t_lo, t_hi = float(t.min()), float(t.max())
        support = float(sum(p.length for p in group))
        if support < 0.19 * h:
            return None
        if abs(d[1]) > self.max_sin:
            return None
        thickness = float(np.median(np.concatenate([np.full(len(p.pixels), p.width) for p in group])))
        p1, p2 = c + d * t_lo, c + d * t_hi

        off = thickness / 2 + 5
        samples = np.linspace(0.05, 0.95, 24)
        side_ok = [0, 0]
        side_cols = []
        for tt in samples:
            q = p1 + (p2 - p1) * tt
            for si, sgn in enumerate((-1, 1)):
                xi, yi = (q + n * sgn * off).round().astype(int)
                if 0 <= xi < w and 0 <= yi < h and region[yi, xi] and not people[yi, xi]:
                    side_ok[si] += 1
                    side_cols.append(lab[yi, xi])
        side_ice = min(side_ok) / len(samples)
        if side_ice < 0.35 or not side_cols:
            return None
        xi, yi = pts[:, 0].astype(int), pts[:, 1].astype(int)
        delta = _trimmed_mean(lab[yi, xi]) - _trimmed_mean(np.array(side_cols))
        if delta[0] > -8:   # paint is darker than the ice
            return None
        p1, p2 = _extend_in_region(p1, p2, region)
        return RinkLine("unknown", p1, p2, thickness, tuple(float(v) for v in delta), support, side_ice)

    @staticmethod
    def _classify(lines: list[RinkLine]) -> list[RinkLine]:
        for ln in lines:
            ln.kind = "center_line" if redness(ln.delta_lab) > RED_THRESHOLD else "blue_line"
        return lines


RED_THRESHOLD = 0.5


def _trimmed_mean(x: np.ndarray, cut: float = 0.2) -> np.ndarray:
    x = np.sort(np.asarray(x, np.float32), axis=0)
    k = int(len(x) * cut)
    return x[k:len(x) - k].mean(0) if len(x) > 2 * k else x.mean(0)


def redness(delta_lab) -> float:
    """Red paint raises a* (and slightly b*) relative to the ice; blue paint lowers b*."""
    return float(delta_lab[1] + delta_lab[2])


def _extend_in_region(p1: np.ndarray, p2: np.ndarray, region: np.ndarray,
                      step: float = 3.0, max_gap: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Extend a segment in both directions while it stays on the ice surface, so a line
    partly hidden by players is drawn boards to boards."""
    h, w = region.shape
    d = (p2 - p1) / (np.linalg.norm(p2 - p1) + 1e-9)

    def walk(p, direction):
        last_in, gap, q = p, 0, p.copy()
        while True:
            q = q + direction * step
            xi, yi = int(round(q[0])), int(round(q[1]))
            if not (0 <= xi < w and 0 <= yi < h):
                return last_in
            if region[yi, xi]:
                last_in, gap = q.copy(), 0
            else:
                gap += 1
                if gap > max_gap:
                    return last_in

    return walk(p1, -d), walk(p2, d)


def segments_cross(a1, a2, b1, b2) -> bool:
    """True if segments a1-a2 and b1-b2 intersect."""
    def orient(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return (orient(a1, a2, b1) != orient(a1, a2, b2)) and (orient(b1, b2, a1) != orient(b1, b2, a2))


def drop_crossing(items: list, key, get=lambda it: (it.p1, it.p2)) -> list:
    """Rink lines are parallel on the ice and never cross in view; when two candidates do,
    keep the stronger one."""
    items = sorted(items, key=key, reverse=True)
    kept: list = []
    for it in items:
        a1, a2 = get(it)
        if not any(segments_cross(a1, a2, *get(k)) for k in kept):
            kept.append(it)
    return kept


def _group_collinear(pieces: list[_Piece], tol: float, ang_tol_deg: float = 6.0) -> list[list[_Piece]]:
    """Greedy grouping of stripe pieces that lie on the same straight line."""
    pieces = sorted(pieces, key=lambda p: -p.length)
    groups: list[list[_Piece]] = []
    cos_tol = np.cos(np.deg2rad(ang_tol_deg))
    for p in pieces:
        for g in groups:
            ref = g[0]
            if abs(float(p.direction @ ref.direction)) < cos_tol:
                continue
            nrm = np.array([-ref.direction[1], ref.direction[0]])
            if abs(float((p.center - ref.center) @ nrm)) < tol + 0.5 * (p.width + ref.width):
                g.append(p)
                break
        else:
            groups.append([p])
    return groups
