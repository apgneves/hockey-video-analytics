"""Puck-focus mode: follow the puck as precisely as possible, plus the rink lines and goals.
No player tracking.

Pass 1 (analyze): the detector runs on every new frame. A puck candidate is kept only if it
  is on the ice, isn't a glass-stanchion clip (the clips on top of the glass posts look just
  like pucks to the detector, but each sits on a thin post running down from it) and isn't a
  screen overlay such as a mouse cursor (pixel-identical on screen while the camera moves).
  Camera motion is chained into per-shot coordinates; lines and goals are tracked as usual.
Pass 2 (solve): per camera shot, candidates at known clip/overlay spots and anything that
  stays put in rink coordinates for a second (fixed equipment) are dropped. The puck path is
  then the longest path through the remaining candidates (at most one per frame, every step
  physically reachable), with short gaps interpolated. Frames near the path where the puck
  wasn't found get a zoomed second look, and the path is solved again.
Pass 3 (render): the video is read again and drawn.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .camera import CameraMotion, warp_points
from .color import apply_gains, ice_fraction, ice_gains
from .detector import CENTER_ICE, GOAL, PERSON_CLASSES, PUCK, DetectorPool, resolve_devices
from .lines import RinkLineDetector
from .pipeline import MIN_ICE_FRACTION, ProgressFn, Settings, on_ice, warp_state
from .render import Renderer
from .tracking import GoalTracker, LineTracker

CANDIDATE_SCORE = 0.06     # low on purpose: the path solver rejects inconsistent candidates
TAU = 0.15                 # detections weaker than this count against the path
MAX_SKIP = 3               # frames a tracklet may go undetected and still continue
STATIC_SECONDS = 1.0       # a "puck" that stays put (in rink terms) this long is fixed equipment
STATIC_RADIUS = 0.006      # (fraction of the frame diagonal) how still "put" is
BRIDGE_SECONDS = 0.4       # longest gap drawn as a straight interpolated puck path
JUMP_SECONDS = 0.3         # the puck may be re-found anywhere after being lost this long...
TELEPORT_PENALTY = 0.3     # ...at this cost (keeps the path from hopping between candidates)
GAP_PENALTY = 0.01         # per-frame cost of a bridged gap
DIST_WEIGHT = 0.1          # prefer the candidate that needs the smallest move
TRAIL = 12                 # frames of trail drawn behind the puck
REFINE_SECONDS = 1.0       # zoomed re-check of frames this close to a known puck position
CLIP_SCORE = 0.43          # glass_post_score at/above this: a glass-stanchion clip (pucks score <0.4)
CLIP_SUSPECT = 0.25        # repeated scores at/above this at one spot also mark a clip
CLIP_RADIUS = 0.012        # (fraction of the frame diagonal) same spot, in shot coordinates


def glass_post_score(gray: np.ndarray, cx: float, cy: float, size: float, max_deg: float = 12.0) -> float:
    """Fraction of a straight, nearly vertical thin dark line running down from (cx, cy).
    ~1.0 for the clips on top of glass stanchions, ~0 for a puck on the ice. The threshold is
    low because distant posts are faint gray lines behind the glass."""
    H, W = gray.shape
    y0 = int(round(cy + 0.9 * size))
    L = min(int(np.clip(6 * size, 50, 90)), H - 1 - y0)
    if L < 25:
        return 0.0
    half = int(L * np.tan(np.deg2rad(max_deg))) + 6
    xa, xb = int(cx) - half, int(cx) + half + 1
    if xa < 0 or xb > W:
        return 0.0
    crop = gray[y0:y0 + L, xa:xb]
    ridge = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)))
    ridge = cv2.dilate(ridge, np.ones((1, 3), np.uint8))       # +-1 px tolerance across the line
    ys = np.arange(L)
    best = 0.0
    for deg in np.arange(-max_deg, max_deg + 0.1, 1.5):
        xs = np.round(cx - xa + ys * np.tan(np.deg2rad(deg))).astype(int)
        ok = (xs >= 0) & (xs < crop.shape[1])
        best = max(best, float((ridge[ys[ok], xs[ok]] > 4).mean()))
    return best


class OverlayCheck:
    """Detects screen overlays (mouse cursor, scorebug, player UI in screen recordings): a patch
    that stays pixel-identical in the image while the camera moves isn't part of the scene.
    A real puck, even one the camera follows, always shifts by at least a few pixels."""

    def __init__(self, history: int = 16, half: int = 12):
        self.frames: deque = deque(maxlen=history)   # (gray, frame -> shot homography, shot)
        self.half = half

    def push(self, gray: np.ndarray, M: np.ndarray, shot: int) -> None:
        self.frames.append((gray, M, shot))

    def is_overlay(self, gray: np.ndarray, M: np.ndarray, shot: int, cx: float, cy: float) -> bool:
        h = self.half
        x, y = int(round(cx)), int(round(cy))
        H, W = gray.shape
        if not (h <= x < W - h and h <= y < H - h):
            return False
        patch = gray[y - h:y + h, x - h:x + h].astype(np.int16)
        if patch.std() < 8:                              # untextured: can't tell
            return False
        for old, M_old, s_old in self.frames:            # oldest first: most camera motion
            if s_old != shot:
                continue
            # where the scene point now at (x, y) was in the old frame
            px, py = warp_points(np.linalg.inv(M_old) @ M, np.array([[cx, cy]]))[0]
            if np.hypot(px - cx, py - cy) < h:           # camera hasn't moved enough to tell
                continue
            return float(np.abs(old[y - h:y + h, x - h:x + h].astype(np.int16) - patch).mean()) < 2.5
        return False


OVERLAY_SECONDS = 3.0      # a confirmed screen overlay also removes candidates at that spot this long before/after
OVERLAY_RADIUS = 12        # pixels


def drop_overlay_spots(cands: dict[int, np.ndarray], overlays: list, fps_u: float) -> dict[int, np.ndarray]:
    """Remove candidates at the screen position of a confirmed overlay (e.g. a mouse cursor)
    within a few seconds of it, including frames before the camera moved enough to prove it."""
    if not overlays:
        return cands
    ov = np.asarray(overlays, np.float64)                  # rows: frame, x, y (image pixels)
    span = OVERLAY_SECONDS * fps_u
    out = {}
    for t, c in cands.items():
        near = ov[np.abs(ov[:, 0] - t) <= span]
        if len(c) and len(near):
            d = np.linalg.norm(c[:, None, 5:7] - near[None, :, 1:3], axis=2).min(1)
            c = c[d > OVERLAY_RADIUS]
        out[t] = c
    return out


def drop_clip_spots(cands: dict[int, np.ndarray], clips: list, diag: float) -> dict[int, np.ndarray]:
    """Remove candidates at a glass-clip spot. Clips are fixed in shot coordinates, so a clip
    rejected in one frame (post clearly visible) also removes it in frames where its post is
    too faint to see. Spots where borderline post scores keep recurring count as clips too."""
    r = CLIP_RADIUS * diag
    spots = [np.asarray(clips, np.float64).reshape(-1, 2)] if len(clips) else []
    susp = np.vstack([c[c[:, 4] >= CLIP_SUSPECT, :2] for c in cands.values()] or [np.zeros((0, 2))])
    if len(susp) >= 3:
        near = cKDTree(susp).query_ball_point(susp, r)
        spots.append(susp[[len(n) >= 3 for n in near]])
    spots = [s for s in spots if len(s)]
    if not spots:
        return cands
    tree = cKDTree(np.vstack(spots))
    out = {}
    for t, c in cands.items():
        d, _ = tree.query(c[:, :2]) if len(c) else (np.zeros(0), None)
        out[t] = c[d > r]
    return out


# --------------------------------------------------------------------------- pass 2: path solver
@dataclass
class Tracklet:
    t: list[int] = field(default_factory=list)          # unique-frame index within the shot
    j: list[int] = field(default_factory=list)          # candidate index within that frame
    p: list[np.ndarray] = field(default_factory=list)   # position in shot coordinates
    v: np.ndarray = field(default_factory=lambda: np.zeros(2))   # velocity per frame

    @property
    def start(self) -> int:
        return self.t[0]

    @property
    def end(self) -> int:
        return self.t[-1]

    def add(self, t: int, j: int, p: np.ndarray) -> None:
        if self.t:
            v = (p - self.p[-1]) / max(t - self.t[-1], 1)
            self.v = v if len(self.t) == 1 else 0.6 * v + 0.4 * self.v
        self.t.append(t)
        self.j.append(j)
        self.p.append(p)


def build_tracklets(cands: dict[int, np.ndarray], n_frames: int, diag: float, fps_u: float,
                    stride: int = 1) -> list[Tracklet]:
    """Link candidates (t -> array of [x, y, size, score] in shot coordinates) frame to frame.
    stride: the detector ran on every stride-th frame, so gaps of that size are normal."""
    base_gate, vmax = 0.02 * diag, 0.06 * diag
    max_skip = MAX_SKIP * stride
    static_keep = int(round(fps_u))            # a motionless tracklet may wait this long for its object
    active: list[Tracklet] = []
    done: list[Tracklet] = []
    for t in range(n_frames):
        c = cands.get(t)
        alive = []
        for tr in active:
            still = np.linalg.norm(tr.v) < 0.004 * diag
            (alive if t - tr.end <= (max(static_keep, max_skip) if still else max_skip) else done).append(tr)
        active = alive
        if c is None or len(c) == 0:
            continue
        used = set()
        if active:
            cost = np.full((len(active), len(c)), 1e6)
            for i, tr in enumerate(active):
                dt = t - tr.end
                pred = tr.p[-1] + tr.v * dt
                gate = base_gate + 0.5 * np.linalg.norm(tr.v) * dt if len(tr.t) > 1 else base_gate + vmax * dt
                d = np.linalg.norm(c[:, :2] - pred, axis=1)
                ok = d <= gate
                cost[i, ok] = d[ok] / gate - c[ok, 3]
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < 1e5:
                    active[i].add(t, int(j), c[j, :2].copy())
                    used.add(j)
        for j in range(len(c)):
            if j not in used:
                tr = Tracklet()
                tr.add(t, j, c[j, :2].copy())
                active.append(tr)
    return done + active


def is_static(tr: Tracklet, diag: float, fps_u: float) -> bool:
    if tr.end - tr.start < STATIC_SECONDS * fps_u:
        return False
    P = np.array(tr.p)
    return float(np.linalg.norm(P - np.median(P, 0), axis=1).max()) < STATIC_RADIUS * diag


def static_candidates(cands: dict[int, np.ndarray], n_frames: int, diag: float, fps_u: float,
                      stride: int = 1) -> set[tuple[int, int]]:
    """(frame, index) of candidates on a motionless object seen for seconds (glass fittings,
    cameras, ice markings). The puck is never that still for that long during play."""
    out: set[tuple[int, int]] = set()
    for tr in build_tracklets(cands, n_frames, diag, fps_u, stride):
        if is_static(tr, diag, fps_u):
            out.update(zip(tr.t, tr.j))
    return out


@dataclass
class PathPoint:
    t: int
    pos: np.ndarray
    size: float
    score: float
    link: str        # how it joins the previous point: 'start' | 'near' (short gap) | 'jump' (re-found)


def solve_path(cands: dict[int, np.ndarray], n_frames: int, diag: float, fps_u: float,
               stride: int = 1) -> list[PathPoint]:
    """Best puck path: at most one candidate per frame, maximizing detector evidence while
    every step stays within how far the puck can travel. Solved exactly as a longest path
    through the candidates (a DAG ordered by time)."""
    static = static_candidates(cands, n_frames, diag, fps_u, stride)
    base_gate, vmax = 0.02 * diag, 0.06 * diag
    bridge = max(2 * stride, int(round(BRIDGE_SECONDS * fps_u)))
    jump_frames = max(stride + 1, int(round(JUMP_SECONDS * fps_u)))
    P, S, Z = {}, {}, {}
    for t, c in cands.items():
        keep = [j for j in range(len(c)) if (t, j) not in static]
        if keep:
            P[t], Z[t], S[t] = c[keep, :2], c[keep, 2], c[keep, 3]
    best: dict[int, np.ndarray] = {}
    back: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    jump_val, jump_ref = -np.inf, (-1, -1)
    waiting: deque = deque()
    for t in sorted(P):
        while waiting and waiting[0] <= t - jump_frames:     # old enough to re-find the puck from
            tp = waiting.popleft()
            i = int(np.argmax(best[tp]))
            if best[tp][i] > jump_val:
                jump_val, jump_ref = float(best[tp][i]), (tp, i)
        k = len(P[t])
        val = np.zeros(k)
        src_t, src_i, link = np.full(k, -1), np.full(k, -1), np.zeros(k, int)   # link 0 start, 1 near, 2 jump
        if jump_val - TELEPORT_PENALTY > 0:
            val[:] = jump_val - TELEPORT_PENALTY
            src_t[:], src_i[:], link[:] = jump_ref[0], jump_ref[1], 2
        for g in range(1, bridge + 1):
            tp = t - g
            if tp not in best:
                continue
            reach = base_gate + vmax * g
            D = np.linalg.norm(P[t][:, None, :] - P[tp][None, :, :], axis=2)
            cand = best[tp][None, :] - GAP_PENALTY * g - DIST_WEIGHT * D / reach
            cand[D > reach] = -np.inf
            i = cand.argmax(1)
            m = cand[np.arange(k), i]
            upd = m > val
            val[upd], src_t[upd], src_i[upd], link[upd] = m[upd], tp, i[upd], 1
        best[t] = S[t] - TAU + val
        back[t] = (src_t, src_i, link)
        waiting.append(t)
    if not best:
        return []
    t = max(best, key=lambda f: best[f].max())
    i = int(np.argmax(best[t]))
    if best[t][i] <= 0:
        return []
    path = []
    while t >= 0:
        st, si, lk = back[t]
        path.append(PathPoint(t, P[t][i], float(Z[t][i]), float(S[t][i]), ("start", "near", "jump")[lk[i]]))
        t, i = int(st[i]), int(si[i])
    return path[::-1]


def path_positions(path: list[PathPoint]) -> dict[int, tuple[np.ndarray, float, float, str]]:
    """Per unique frame: (position, size, score, 'detected' | 'interpolated'). Short gaps
    between consecutive path points are filled in a straight line."""
    out: dict[int, tuple[np.ndarray, float, float, str]] = {}
    prev = None
    for pt in path:
        if prev is not None and pt.link == "near":
            for t in range(prev.t + 1, pt.t):
                w = (t - prev.t) / (pt.t - prev.t)
                out[t] = ((1 - w) * prev.pos + w * pt.pos, (1 - w) * prev.size + w * pt.size, 0.0, "interpolated")
        out[pt.t] = (pt.pos, pt.size, pt.score, "detected")
        prev = pt
    return out


# --------------------------------------------------------------------------- driver
@dataclass
class _Info:
    idx: int
    shot: int = -1                 # -1: no rink in view
    M: np.ndarray | None = None    # this frame -> shot coordinates
    uid: int = -1                  # new-frame index within the shot (repeated frames share it)
    goals: list | None = None
    lines: list | None = None
    same_as: "_Info | None" = None  # repeated frame: annotations come from this one


class PuckFocusProcessor:
    def __init__(self, settings: Settings | None = None):
        self.cfg = settings or Settings(detect_every=1)
        self.devices = resolve_devices(self.cfg.device)
        cpu_threads = max(2, (os.cpu_count() or 4) // 2 - 2)
        self.pool = DetectorPool(self.devices, self.cfg.weights, self.cfg.imgsz, cpu_threads)

    def close(self) -> None:
        self.pool.close()

    def process(self, video_path: str | Path, out_video: str | Path | None, out_json: str | Path | None = None,
                progress: ProgressFn | None = None) -> dict:
        cfg = self.cfg
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = int(round(cfg.start_s * fps))
        end = n_total if cfg.max_frames is None else min(n_total, start + cfg.max_frames)
        n_frames = max(0, end - start)
        stride = max(1, int(cfg.detect_every))
        diag = float(np.hypot(W, H))
        report = progress or (lambda i, n, msg: None)
        t0 = time.time()

        infos, cands, clips, overlays, stats = self._analyze(cap, start, n_frames, stride, W, report)
        cap.release()

        # ---- pass 2: solve the puck path shot by shot, then take a zoomed second look at the
        # frames near the path where the puck wasn't found, and solve again
        uniq_per_shot: dict[int, int] = {}
        for inf in infos:
            if inf.shot >= 0:
                uniq_per_shot[inf.shot] = max(uniq_per_shot.get(inf.shot, 0), inf.uid + 1)
        n_unique = sum(uniq_per_shot.values())
        play_frames = sum(1 for i in infos if i.shot >= 0)
        fps_u = n_unique / max(play_frames / fps, 1e-6) if play_frames else fps   # new frames per second

        def solve_all() -> dict[int, dict]:
            report(0, 1, "Solving puck path")
            out = {}
            for shot, n_u in uniq_per_shot.items():
                shot_cands = drop_clip_spots({t: c for (s, t), c in cands.items() if s == shot},
                                             clips.get(shot, []), diag)
                shot_cands = drop_overlay_spots(shot_cands, overlays.get(shot, []), fps_u)
                out[shot] = path_positions(solve_path(shot_cands, n_u, diag, fps_u, stride))
            return out

        paths = solve_all()
        if cfg.refine_puck:
            stats["zoom_rechecks"], stats["zoom_candidates"] = self._refine(video_path, start, infos, cands, clips, overlays,
                                                                            paths, fps_u, report)
            paths = solve_all()

        # ---- pass 3: render
        summary = self._render(video_path, start, infos, paths, fps, W, H, out_video, out_json, report, t0,
                               stats | {"unique_frames": n_unique})
        report(n_frames, n_frames, "Done")
        return summary

    # -------------------------------------------------------------- pass 1
    def _analyze(self, cap, start: int, n_frames: int, stride: int, W: int, report: ProgressFn):
        cfg = self.cfg
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        camera = CameraMotion()
        line_det, line_trk, goal_trk = RinkLineDetector(), LineTracker(W), GoalTracker()
        infos: list[_Info] = []
        cands: dict[tuple[int, int], np.ndarray] = {}   # (shot, frame) -> [x, y, size, score, post]
        clips: dict[int, list] = {}                       # shot -> glass-clip positions (shot coords)
        overlays: dict[int, list] = {}                    # shot -> (frame, x, y) of screen overlays (image)
        queue: deque = deque()          # (info, frame, future | None), processed in order
        state = {"shot": None, "M": None, "lines": [], "goals": [], "boxes": np.zeros((0, 4), np.float32)}
        stats = {"detector_devices": self.pool.names, "duplicate_frames": 0, "non_play_frames": 0, "shots": 0,
                 "puck_candidates": 0, "rejected_glass_clips": 0, "rejected_overlays": 0}
        overlay = OverlayCheck()
        t0 = time.time()

        def process(info: _Info, frame: np.ndarray, fut: Future | None) -> None:
            if info.shot != state["shot"]:
                line_trk.reset()
                goal_trk.reset()
                state.update(shot=info.shot, M=info.M, lines=[], goals=[])
            Hrel = np.linalg.inv(info.M) @ state["M"]      # previous processed frame -> this frame
            state["M"] = info.M
            if fut is None:                                # detector skipped (detect_every > 1)
                moved = warp_state({"players": [], "puck": None, "goals": state["goals"], "lines": state["lines"]}, Hrel)
                info.lines, info.goals = moved["lines"], moved["goals"]
                state.update(lines=info.lines, goals=info.goals)
                return
            det = fut.result()
            persons = det.select(*PERSON_CLASSES, min_score=0.3)
            state["boxes"] = persons.boxes
            lines = line_trk.update(line_det.detect(frame, persons.boxes), Hrel, det.select(CENTER_ICE, min_score=0.5).boxes)
            region, s = line_det.debug["region"], line_det.debug["scale"]
            g = det.select(GOAL, min_score=cfg.goal_min_score)
            goals = goal_trk.update(g.boxes, g.scores, Hrel)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kept = []
            pk = det.select(PUCK, min_score=CANDIDATE_SCORE)
            for b, sc in zip(pk.boxes, pk.scores):
                cx, cy, size = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, float(max(b[2] - b[0], b[3] - b[1]))
                if size > 0.06 * frame.shape[0] or not on_ice(region, s, b):
                    continue
                stats["puck_candidates"] += 1
                if overlay.is_overlay(gray, info.M, info.shot, cx, cy):
                    stats["rejected_overlays"] += 1
                    overlays.setdefault(info.shot, []).append((info.uid, cx, cy))
                    continue
                X, Y = warp_points(info.M, np.array([[cx, cy]]))[0]
                post = glass_post_score(gray, cx, cy, size)
                if post >= CLIP_SCORE:
                    stats["rejected_glass_clips"] += 1
                    clips.setdefault(info.shot, []).append((X, Y))
                    continue
                kept.append((X, Y, size, float(sc), post, cx, cy))
            cands[(info.shot, info.uid)] = np.array(kept, np.float64).reshape(-1, 7)
            overlay.push(gray, info.M, info.shot)
            info.lines, info.goals = lines, goals
            state.update(lines=lines, goals=goals)

        def drain(final: bool = False) -> None:
            while queue:
                info, frame, fut = queue[0]
                in_flight = sum(1 for q in queue if q[2] is not None)
                if not (final or fut is None or fut.done() or in_flight > cfg.lookahead):
                    return
                queue.popleft()
                process(info, frame, fut)

        shot, M, uid, new_shot = -1, np.eye(3), -1, True
        prev_unique: _Info | None = None
        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            fi = start + i
            gains = ice_gains(frame)
            motion = camera.update(frame, state["boxes"])
            if ice_fraction(frame, gains) < MIN_ICE_FRACTION:
                infos.append(_Info(fi))
                stats["non_play_frames"] += 1
                new_shot = True
            elif motion.duplicate and not new_shot and prev_unique is not None:
                infos.append(_Info(fi, shot, M, uid, same_as=prev_unique))
                stats["duplicate_frames"] += 1
            else:
                if motion.scene_cut or new_shot:
                    shot, M, uid, new_shot = shot + 1, np.eye(3), -1, False
                    stats["shots"] += 1
                else:
                    M = M @ np.linalg.inv(motion.H)
                uid += 1
                info = _Info(fi, shot, M, uid)
                infos.append(info)
                prev_unique = info
                fut = self.pool.submit(apply_gains(frame, gains)) if uid % stride == 0 else None
                queue.append((info, frame, fut))
            drain()
            if (i + 1) % 10 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                report(i + 1, n_frames, f"Analyzing frame {i + 1}/{n_frames} ({rate:.1f} frames/s, "
                                        f"~{(n_frames - i - 1) / rate / 60:.1f} min left)")
        drain(final=True)
        for inf in infos:
            if inf.same_as is not None:
                inf.goals, inf.lines = inf.same_as.goals, inf.same_as.lines
        return infos, cands, clips, overlays, stats

    # -------------------------------------------------------------- pass 2b
    def _refine(self, video_path: Path, start: int, infos: list[_Info], cands: dict, clips: dict[int, list],
                overlays: dict[int, list],
                paths: dict[int, dict], fps_u: float, report: ProgressFn) -> tuple[int, int]:
        """Zoomed re-detection in frames close (in time) to the solved path where no puck was
        placed: crop around where the puck last was, enlarge, detect again. The puck is small
        enough that the extra magnification recovers some of the misses."""
        reach = max(2, int(round(REFINE_SECONDS * fps_u)))
        targets: dict[int, np.ndarray] = {}           # index into infos -> reference position (shot coords)
        for n, inf in enumerate(infos):
            if inf.shot < 0 or inf.same_as is not None:
                continue
            path = paths.get(inf.shot, {})
            if inf.uid in path or not path:
                continue
            near = [t for t in range(inf.uid - reach, inf.uid + reach + 1) if t in path]
            if near:
                targets[n] = path[min(near, key=lambda t: (abs(t - inf.uid), t > inf.uid))][0]
        if not targets:
            return 0, 0
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        queue: deque = deque()
        overlay = OverlayCheck()
        added = 0

        def collect(n: int, gray: np.ndarray, x0: int, y0: int, fut: Future) -> int:
            inf, det = infos[n], fut.result().select(PUCK, min_score=CANDIDATE_SCORE)
            kept = []
            for b, sc in zip(det.boxes, det.scores):
                bx = b + np.array([x0, y0, x0, y0], np.float32)          # crop pixels -> full-frame pixels
                cx, cy, size = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2, float(max(bx[2] - bx[0], bx[3] - bx[1]))
                if size > 0.06 * gray.shape[0]:
                    continue
                if overlay.is_overlay(gray, inf.M, inf.shot, cx, cy):
                    overlays.setdefault(inf.shot, []).append((inf.uid, cx, cy))
                    continue
                X, Y = warp_points(inf.M, np.array([[cx, cy]]))[0]
                post = glass_post_score(gray, cx, cy, size)
                if post >= CLIP_SCORE:
                    clips.setdefault(inf.shot, []).append((X, Y))
                    continue
                kept.append((X, Y, size, float(sc), post, cx, cy))
            if kept:
                key = (inf.shot, inf.uid)
                old = cands.get(key, np.zeros((0, 7)))
                cands[key] = np.vstack([old, np.array(kept)])
            return len(kept)

        last = max(targets)
        for n in range(last + 1):
            ok, frame = cap.read()
            if not ok:
                break
            inf = infos[n]
            if inf.shot >= 0 and inf.same_as is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                overlay.push(gray, inf.M, inf.shot)
            if n in targets:
                c = warp_points(np.linalg.inv(inf.M), targets[n][None])[0]
                H, W = frame.shape[:2]
                cw, ch = W // 2, H // 2
                x0 = int(np.clip(c[0] - cw / 2, 0, W - cw))
                y0 = int(np.clip(c[1] - ch / 2, 0, H - ch))
                # the detector scales its input to imgsz wide, so a half-width crop is seen at 2x the
                # magnification of the full frame
                crop = apply_gains(frame[y0:y0 + ch, x0:x0 + cw], ice_gains(frame))
                queue.append((n, gray, x0, y0, self.pool.submit(crop)))
            while queue and (queue[0][-1].done() or len(queue) > self.cfg.lookahead):
                added += collect(*queue.popleft())
            if (n + 1) % 25 == 0:
                report(n + 1, last + 1, f"Zoomed re-check {n + 1}/{last + 1}")
        while queue:
            added += collect(*queue.popleft())
        cap.release()
        return len(targets), added

    # -------------------------------------------------------------- pass 3
    def _render(self, video_path: Path, start: int, infos: list[_Info], paths: dict[int, dict], fps: float,
                W: int, H: int, out_video, out_json, report: ProgressFn, t0: float, stats: dict) -> dict:
        cfg = self.cfg
        renderer = Renderer(W, H, [], [], puck_focus=True)
        writer = None
        if out_video is not None and cfg.render_video:
            Path(out_video).parent.mkdir(parents=True, exist_ok=True)
            writer = imageio_ffmpeg.write_frames(
                str(out_video), (W, H), fps=fps, codec="libx264", pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
                macro_block_size=1, quality=None, ffmpeg_log_level="error",
                output_params=["-crf", "21", "-preset", "veryfast", "-movflags", "+faststart"])
            writer.send(None)
        jf = None
        if out_json is not None:
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            jf = open(out_json, "w", encoding="utf-8")
            jf.write('{"frames": [')
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        counts = {"frames_with_puck_detected": 0, "frames_with_puck_interpolated": 0, "frames_with_goal": 0,
                  "frames_with_blue_line": 0, "frames_with_center_line": 0}
        t_r = time.time()
        try:
            for n, inf in enumerate(infos):
                ok, frame = cap.read()
                if not ok:
                    break
                puck, rec_puck = None, None
                goals, lines = inf.goals or [], inf.lines or []
                if inf.shot >= 0:
                    path = paths.get(inf.shot, {})
                    hit = path.get(inf.uid)
                    if hit is not None:
                        Minv = np.linalg.inv(inf.M)
                        pos, size, score, kind = hit
                        center = warp_points(Minv, pos[None])[0]
                        trail_ref = [path[t][0] for t in range(inf.uid - TRAIL, inf.uid) if t in path]
                        trail = list(warp_points(Minv, np.array(trail_ref))) if trail_ref else []
                        puck = {"center": center, "size": size, "score": score, "predicted": kind != "detected",
                                "trail": trail}
                        rec_puck = {"center": [round(float(v), 1) for v in center], "state": kind,
                                    "score": round(float(score), 3)}
                        counts["frames_with_puck_detected" if kind == "detected" else "frames_with_puck_interpolated"] += 1
                counts["frames_with_goal"] += int(bool(goals))
                counts["frames_with_blue_line"] += int(any(l["kind"] == "blue_line" for l in lines))
                counts["frames_with_center_line"] += int(any(l["kind"] == "center_line" for l in lines))
                if writer is not None:
                    img = renderer.draw(frame, [], puck, goals, lines) if inf.shot >= 0 else frame
                    writer.send(np.ascontiguousarray(img))
                if jf is not None:
                    rec = {"frame": inf.idx, "time": round(inf.idx / fps, 3), "play": inf.shot >= 0, "puck": rec_puck,
                           "goals": [{"box": [round(float(v), 1) for v in g["box"]], "score": round(g["score"], 3)} for g in goals],
                           "lines": [{"kind": l["kind"], "p1": [round(float(v), 1) for v in l["p1"]],
                                      "p2": [round(float(v), 1) for v in l["p2"]]} for l in lines]}
                    jf.write(("\n" if n == 0 else ",\n") + json.dumps(rec, separators=(",", ":")))
                if (n + 1) % 25 == 0:
                    rate = (n + 1) / max(time.time() - t_r, 1e-6)
                    report(n + 1, len(infos), f"Rendering frame {n + 1}/{len(infos)} ({rate:.0f} frames/s)")
            elapsed = time.time() - t0
            n = len(infos)
            summary = {"mode": "puck_focus", "video": str(video_path),
                       "output_video": str(out_video) if writer is not None else None,
                       "fps": fps, "width": W, "height": H, "frames_processed": n,
                       "processing_seconds": round(elapsed, 1), "processing_fps": round(n / max(elapsed, 1e-6), 2),
                       "detect_every": max(1, int(cfg.detect_every))} | stats | counts
            summary["frames_with_puck"] = counts["frames_with_puck_detected"] + counts["frames_with_puck_interpolated"]
        except BaseException:
            summary = {"error": "processing stopped before the end of the video"}
            raise
        finally:
            cap.release()
            if writer is not None:
                writer.close()
            if jf is not None:
                jf.write('\n],\n"summary": ' + json.dumps(summary, indent=1) + "\n}\n")
                jf.close()
        return summary
