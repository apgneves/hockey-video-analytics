"""End-to-end video processing: detect, track, classify and render.

How it stays fast:
  * The detector (by far the most expensive step) only runs on keyframes: every Nth *new*
    frame. Repeated frames (common in padded or screen-recorded video) don't count. Frames
    between keyframes get positions interpolated from the surrounding keyframes, corrected
    for camera motion, which is measured on every frame.
  * Detection runs in background worker threads (GPU and/or CPU, see DetectorPool) several
    keyframes ahead, while the main thread tracks, finds rink lines and renders.
  * Stretches with no rink in view (crowd, intermission, studio shots) skip detection.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import imageio_ffmpeg
import numpy as np

from .camera import CameraMotion, Displacement, FlowAccumulator, warp_points
from .color import apply_gains, ice_fraction, ice_gains
from .detector import (CENTER_ICE, GOAL, PERSON_CLASSES, PLAYER, PUCK, REFEREE, DetectorPool, Detections,
                       resolve_devices)
from .lines import RinkLineDetector
from .render import Renderer
from .teams import TeamClassifier, is_colored_jersey, torso_feature
from .tracking import GoalTracker, LineTracker, PlayerTracker, PuckTracker

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "HockeyAI.pt"
MIN_ICE_FRACTION = 0.12   # below this the rink isn't in view (see color.ice_fraction)


@dataclass
class Settings:
    weights: Path = DEFAULT_WEIGHTS
    imgsz: int = 1280            # detector input size; 960 is ~2x faster but finds fewer pucks
    device: str = "auto"         # auto | cpu | dml | cuda, or a list such as "dml,cpu"
    detect_every: int = 3        # run the detector on every Nth new frame (1 = every frame)
    render_video: bool = True    # False: only write the JSON (faster for whole games)
    puck_min_score: float = 0.12
    goal_min_score: float = 0.25   # goal tracker also requires consistent hits
    team_sample_frames: int = 16
    max_frames: int | None = None
    start_s: float = 0.0
    lookahead: int = 4           # keyframes queued ahead so every detector worker stays busy
    refine_puck: bool = True     # puck-focus mode: zoomed re-check of frames near the puck path


ProgressFn = Callable[[int, int, str], None]
_EYE = np.eye(3)
_EMPTY = {"players": [], "puck": None, "goals": [], "lines": []}


@dataclass
class _Frame:
    idx: int
    image: np.ndarray
    H: np.ndarray = field(default_factory=lambda: _EYE)   # camera motion: previous keyframe -> this frame
    A: np.ndarray = field(default_factory=lambda: _EYE)   # similarity approximation of H (player tracker)
    step: int = 0                                          # new frames since the previous keyframe
    flow: Displacement | None = None                       # keyframes: per-pixel motion since the previous one


@dataclass
class _Segment:
    """Frames up to and including a keyframe. key=None: frames with no keyframe after them
    (before a scene cut or a no-rink stretch, or at the end); they are extrapolated."""
    frames: list[_Frame]
    key: _Frame | None = None
    det: Future | None = None
    norm: np.ndarray | None = None
    reset: bool = False
    passthrough: bool = False     # no rink in view: output unannotated


def _fix_referees(norm_frame: np.ndarray, persons: Detections) -> None:
    """The detector sometimes calls dark-jerseyed skaters referees. Referees wear black and
    white stripes, so a 'referee' with a clearly colored torso is relabeled as a player."""
    for i, label in enumerate(persons.labels):
        if label == REFEREE and is_colored_jersey(torso_feature(norm_frame, persons.boxes[i])):
            persons.labels[i] = PLAYER


def _is_glass_clip(gray: np.ndarray, box: np.ndarray) -> bool:
    """The clips on top of glass stanchions look like pucks; see puck_focus.glass_post_score."""
    from .puck_focus import CLIP_SCORE, glass_post_score
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return glass_post_score(gray, cx, cy, float(max(box[2] - box[0], box[3] - box[1]))) >= CLIP_SCORE


def on_ice(region: np.ndarray, scale: float, box: np.ndarray) -> bool:
    """True if the box center lies on the detected ice surface."""
    cx, cy = (box[0] + box[2]) / 2 * scale, (box[1] + box[3]) / 2 * scale
    h, w = region.shape
    xi, yi = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
    r = 3
    patch = region[max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1]
    return bool(patch.mean() > 0.5) if patch.size else False


# --------------------------------------------------------------------------- interpolation
def _warp_box(b: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = warp_points(H, np.array([[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]]))
    return np.array([*pts.min(0), *pts.max(0)], np.float32)


def warp_state(st: dict | None, H: np.ndarray) -> dict | None:
    """Move a keyframe's annotations into another frame's coordinates (camera motion H)."""
    if st is None or np.allclose(H, _EYE):
        return st
    puck = st["puck"]
    if puck is not None:
        pts = warp_points(H, np.vstack([puck["center"][None]] + [np.asarray(puck["trail"]).reshape(-1, 2)]))
        puck = dict(puck, center=pts[0], trail=list(pts[1:]))
    lines = []
    for ln in st["lines"]:
        p1, p2 = warp_points(H, np.stack([ln["p1"], ln["p2"]]))
        lines.append(dict(ln, p1=p1, p2=p2))
    return {"players": [dict(p, box=_warp_box(p["box"], H)) for p in st["players"]], "puck": puck,
            "goals": [dict(g, box=_warp_box(g["box"], H)) for g in st["goals"]], "lines": lines}


def blend_states(a: dict | None, b: dict | None, w: float) -> dict | None:
    """Annotations for a frame between keyframe states a (w=0) and b (w=1), both already
    warped into that frame. Objects seen in both are interpolated; others appear/disappear
    at the midpoint."""
    if a is None or b is None:
        return b if a is None else a
    near = a if w < 0.5 else b
    pa, pb = {p["id"]: p for p in a["players"]}, {p["id"]: p for p in b["players"]}
    near_players = pa if w < 0.5 else pb
    players = []
    for pid in pa.keys() | pb.keys():
        if pid in pa and pid in pb:
            players.append(dict(pb[pid], box=(1 - w) * pa[pid]["box"] + w * pb[pid]["box"]))
        elif pid in near_players:
            players.append(near_players[pid])
    puck = near["puck"]
    if a["puck"] is not None and b["puck"] is not None:
        puck = dict(near["puck"], center=(1 - w) * a["puck"]["center"] + w * b["puck"]["center"],
                    trail=list(a["puck"]["trail"]) + [a["puck"]["center"]])
    return {"players": players, "puck": puck, "goals": near["goals"], "lines": near["lines"]}


# --------------------------------------------------------------------------- per-video state
class _Run:
    """Trackers, renderer and outputs for one video; consumes segments in order."""

    def __init__(self, cfg: Settings, teams: TeamClassifier, fps: float, W: int, H: int,
                 stride: int, out_video: Path | None, out_json: Path | None, report: ProgressFn, n_frames: int):
        self.cfg, self.teams, self.fps, self.report, self.n_frames = cfg, teams, fps, report, n_frames
        self.stride = stride
        self.players = PlayerTracker(updates_per_s=fps / stride, match_buffer=0.15 * (stride - 1))
        self.puck = PuckTracker(frame_diag=float(np.hypot(W, H)), max_missing=max(3, round(8 / stride)),
                                trail=max(6, 20 // stride), gate_frac=0.10 * (1 + 0.35 * (stride - 1)))
        self.goals = GoalTracker(max_missing=max(4, 12 // stride))
        self.line_det = RinkLineDetector()
        self.line_trk = LineTracker(W, max_missing=max(4, 10 // stride), min_hits=3 if stride == 1 else 2)
        self.renderer = Renderer(W, H, teams.team_colors, teams.team_names)
        self.cls_votes: dict[int, Counter] = defaultdict(Counter)
        self.last_state: dict | None = None
        self.last_boxes = np.zeros((0, 4), np.float32)
        self.track_ids: set[int] = set()
        self.stats = Counter()
        # frames are streamed to disk as they're produced (a whole game doesn't fit in memory);
        # finish() appends the summary so the file is one valid JSON object
        self.json = None
        if out_json is not None:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            self.json = open(out_json, "w", encoding="utf-8")
            self.json.write('{"frames": [')
        self.t0 = time.time()
        self.writer = None
        if out_video is not None:
            out_video.parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio_ffmpeg.write_frames(
                str(out_video), (W, H), fps=fps, codec="libx264", pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
                macro_block_size=1, quality=None, ffmpeg_log_level="error",
                output_params=["-crf", "21", "-preset", "veryfast", "-movflags", "+faststart"])
            self.writer.send(None)

    def reset(self) -> None:
        for trk in (self.players, self.puck, self.goals, self.line_trk):
            trk.reset()
        self.last_state = None
        self.stats["scene_cuts"] += 1

    def process(self, seg: _Segment) -> None:
        if seg.passthrough:
            for f in seg.frames:
                self._emit(f, None, play=False)
            self.last_state = None
            return
        if seg.key is None:
            for f in seg.frames:
                self._emit(f, warp_state(self.last_state, f.H))
            return
        det = seg.det.result()
        if seg.reset:
            self.reset()
        key_state = self._update_trackers(seg.key, det, seg.norm)
        to_key = np.linalg.inv(seg.key.H)
        for f in seg.frames:
            w = f.step / max(seg.key.step, 1)
            self._emit(f, blend_states(warp_state(self.last_state, f.H), warp_state(key_state, f.H @ to_key), w))
        self._emit(seg.key, key_state, keyframe=True)
        self.last_state = key_state

    def _update_trackers(self, key: _Frame, det: Detections, norm: np.ndarray) -> dict:
        frame = key.image
        self.stats["keyframes"] += 1
        persons = det.select(*PERSON_CLASSES, min_score=0.12)
        _fix_referees(norm, persons)
        self.last_boxes = persons.boxes[persons.scores > 0.3]

        tracks = self.players.update(persons.boxes, persons.scores, persons.labels, key.A[:2], frame,
                                     flow=key.flow if self.stride > 1 else None)
        for t in tracks:
            self.cls_votes[t["id"]][t["cls"]] += t["score"]
            t["cls"] = self.cls_votes[t["id"]].most_common(1)[0][0]
        skaters = [t for t in tracks if t["cls"] != "referee"]
        for t, team in zip(skaters, self.teams.predict(norm, [t["box"] for t in skaters])):
            t["team"] = self.teams.vote(t["id"], team, t["score"])
        for t in tracks:
            t.setdefault("team", None)

        # rink lines (also gives us the ice surface for the puck check)
        center_ice = det.select(CENTER_ICE, min_score=0.5).boxes
        lines = self.line_trk.update(self.line_det.detect(frame, persons.boxes), key.H, center_ice)
        region, s = self.line_det.debug["region"], self.line_det.debug["scale"]

        pk = det.select(PUCK, min_score=self.cfg.puck_min_score)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cands = [(b, float(sc)) for b, sc in zip(pk.boxes, pk.scores) if on_ice(region, s, b)
                 and not _is_glass_clip(gray, b)]
        puck = self.puck.update(cands, key.H)

        g = det.select(GOAL, min_score=self.cfg.goal_min_score)
        goals = self.goals.update(g.boxes, g.scores, key.H)
        return {"players": tracks, "puck": puck, "goals": goals, "lines": lines}

    def _emit(self, f: _Frame, state: dict | None, keyframe: bool = False, play: bool = True) -> None:
        st = state or _EMPTY
        if self.writer is not None:
            img = self.renderer.draw(f.image, st["players"], st["puck"], st["goals"], st["lines"]) if play else f.image
            self.writer.send(np.ascontiguousarray(img))
        s = self.stats
        s["frames"] += 1
        s["non_play_frames"] += int(not play)
        s["puck_frames"] += int(st["puck"] is not None and not st["puck"]["predicted"])
        s["goal_frames"] += int(bool(st["goals"]))
        s["blue_line_frames"] += int(any(l["kind"] == "blue_line" for l in st["lines"]))
        s["center_line_frames"] += int(any(l["kind"] == "center_line" for l in st["lines"]))
        self.track_ids.update(p["id"] for p in st["players"] if p["cls"] != "referee")
        if self.json is not None:
            rec = json.dumps(_frame_record(f.idx, self.fps, st, keyframe, play), separators=(",", ":"))
            self.json.write(("\n" if s["frames"] == 1 else ",\n") + rec)
        done = s["frames"]
        if done % 10 == 0 or done == self.n_frames:
            rate = done / max(time.time() - self.t0, 1e-6)
            self.report(done, self.n_frames,
                        f"Frame {done}/{self.n_frames} ({rate:.1f} frames/s, ~{(self.n_frames - done) / rate / 60:.1f} min left)")

    def close(self, summary: dict | None = None) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.json is not None:
            self.json.write('\n],\n"summary": ' + json.dumps(summary or {}, indent=1) + "\n}\n")
            self.json.close()
            self.json = None


# --------------------------------------------------------------------------- driver
class VideoProcessor:
    def __init__(self, settings: Settings | None = None):
        self.cfg = settings or Settings()
        self.devices = resolve_devices(self.cfg.device)
        cpu_threads = max(2, (os.cpu_count() or 4) // 2 - 2)   # leave cores for the main thread
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
        report = progress or (lambda i, n, msg: None)
        t_start = time.time()

        # ---- pass 1: learn the two jersey colors from frames spread across the whole video
        teams = TeamClassifier()
        samples = []
        for fi in np.unique(np.linspace(0, n_total - 1, num=cfg.team_sample_frames, dtype=int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if ok:
                norm = apply_gains(frame, ice_gains(frame))
                samples.append((norm, self.pool.submit(norm)))
        for k, (norm, fut) in enumerate(samples):
            report(k, len(samples), "Learning team colors")
            teams.add_samples(norm, fut.result().select(PLAYER, min_score=0.5).boxes)
        teams.fit()

        # ---- pass 2: keyframes are detected in the background; everything else streams
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        run = _Run(cfg, teams, fps, W, H, stride, Path(out_video) if (out_video and cfg.render_video) else None,
                   Path(out_json) if out_json else None, report, n_frames)
        camera = CameraMotion()
        flow = FlowAccumulator() if stride > 1 else None   # per-player motion between keyframes
        segments: deque[_Segment] = deque()
        pending: list[_Frame] = []
        H_acc, A_acc, step = _EYE, _EYE, 0
        need_reset = True

        def key_segment(frames: list[_Frame], key: _Frame, gains: np.ndarray, reset: bool = False) -> _Segment:
            norm = apply_gains(key.image, gains)
            return _Segment(frames, key, self.pool.submit(norm), norm, reset)

        def drain(final: bool = False) -> None:
            """Process finished segments in order; block only when too many keyframes are queued."""
            while segments:
                head = segments[0]
                in_flight = sum(1 for s in segments if s.det is not None)
                if not (final or head.det is None or head.det.done() or in_flight > cfg.lookahead):
                    return
                run.process(segments.popleft())

        try:
            for i in range(n_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                fi = start + i
                gains = ice_gains(frame)
                motion = camera.update(frame, run.last_boxes)
                run.stats["duplicate_frames"] += int(motion.duplicate)
                if flow is not None:
                    flow.update(frame, motion.duplicate)
                if ice_fraction(frame, gains) < MIN_ICE_FRACTION:
                    if pending:
                        segments.append(_Segment(pending))
                        pending = []
                    segments.append(_Segment([_Frame(fi, frame)], passthrough=True))
                    need_reset = True
                    if flow is not None:
                        flow.take()
                elif motion.scene_cut or need_reset:
                    if pending:
                        segments.append(_Segment(pending))
                        pending = []
                    segments.append(key_segment([], _Frame(fi, frame), gains, reset=True))
                    H_acc, A_acc, step, need_reset = _EYE, _EYE, 0, False
                    if flow is not None:
                        flow.take()
                else:
                    H_acc = motion.H @ H_acc
                    A_acc = np.vstack([motion.affine, [0, 0, 1]]) @ A_acc
                    step += int(not motion.duplicate)
                    item = _Frame(fi, frame, H_acc, A_acc, step)
                    if step >= stride:
                        item.flow = flow.take() if flow is not None else None
                        segments.append(key_segment(pending, item, gains))
                        pending = []
                        H_acc, A_acc, step = _EYE, _EYE, 0
                    else:
                        pending.append(item)
                drain()
            if pending:
                segments.append(_Segment(pending))
            drain(final=True)
        except BaseException:
            run.close({"error": "processing stopped before the end of the video"})
            raise
        finally:
            cap.release()

        stats = run.stats
        elapsed = time.time() - t_start
        summary = {
            "video": str(video_path), "output_video": str(out_video) if run.writer else None,
            "fps": fps, "width": W, "height": H, "frames_processed": stats["frames"],
            "processing_seconds": round(elapsed, 1),
            "processing_fps": round(stats["frames"] / max(elapsed, 1e-6), 2),
            "detector_devices": self.pool.names, "detect_every": stride,
            "keyframes": stats["keyframes"], "duplicate_frames": stats["duplicate_frames"],
            "non_play_frames": stats["non_play_frames"],
            "teams": [{"team": k, "color_name": n, "color_bgr": list(c)}
                      for k, (n, c) in enumerate(zip(teams.team_names, teams.team_colors))],
            "unique_player_tracks": len(run.track_ids),
            "frames_with_puck": stats["puck_frames"], "frames_with_goal": stats["goal_frames"],
            "frames_with_blue_line": stats["blue_line_frames"],
            "frames_with_center_line": stats["center_line_frames"],
            "scene_cuts": max(0, stats["scene_cuts"] - 1),   # the first keyframe also resets
        }
        run.close(summary)
        report(n_frames, n_frames, "Done")
        return summary


def _r(v, nd=1):
    return [round(float(x), nd) for x in v]


def _frame_record(fi: int, fps: float, st: dict, keyframe: bool, play: bool) -> dict:
    puck = st["puck"]
    return {
        "frame": fi, "time": round(fi / fps, 3), "keyframe": keyframe, "play": play,
        "players": [{"id": t["id"], "cls": t["cls"], "team": t["team"], "box": _r(t["box"]),
                     "score": round(t["score"], 3)} for t in st["players"]],
        "puck": None if puck is None else {"center": _r(puck["center"]), "score": round(puck["score"], 3),
                                           "predicted": puck["predicted"]},
        "goals": [{"box": _r(g["box"]), "score": round(g["score"], 3)} for g in st["goals"]],
        "lines": [{"kind": l["kind"], "p1": _r(l["p1"]), "p2": _r(l["p2"])} for l in st["lines"]],
    }
