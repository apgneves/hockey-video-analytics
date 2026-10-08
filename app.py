"""Local web UI: upload a hockey MP4, get back an annotated video plus detection data.

    python app.py            # then open http://127.0.0.1:7860
"""
from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from hockey_tracker.pipeline import DEFAULT_WEIGHTS, Settings, VideoProcessor
from hockey_tracker.puck_focus import PuckFocusProcessor

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "web"
SAMPLES = sorted((ROOT / "videos").glob("*.mp4"))

MODES = ["Full analysis: players, teams, puck, lines, goals",
         "Puck focus: puck, lines and goals only, maximum puck accuracy (slower)"]

_processors: dict[tuple[bool, int], VideoProcessor | PuckFocusProcessor] = {}


def _processor(puck_focus: bool, imgsz: int):
    """One processor (and detector worker pool) per mode and input size, reused across runs."""
    key = (puck_focus, imgsz)
    if key not in _processors:
        cls = PuckFocusProcessor if puck_focus else VideoProcessor
        _processors[key] = cls(Settings(imgsz=imgsz))
    return _processors[key]


def analyze(video_path: str | None, mode: str, detect_every: float, fast: bool, data_only: bool,
            max_seconds: float, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload an MP4 first.")
    puck_focus = mode == MODES[1]
    proc = _processor(puck_focus, 960 if fast else 1280)
    proc.cfg.detect_every = 1 if puck_focus else int(detect_every)
    proc.cfg.render_video = not data_only
    proc.cfg.max_frames = None
    if max_seconds and max_seconds > 0:
        import cv2
        fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS) or 30.0
        proc.cfg.max_frames = int(max_seconds * fps)

    stem = Path(video_path).stem
    run_dir = OUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{stem}"
    suffix = "_puck" if puck_focus else "_annotated"
    out_video = None if data_only else run_dir / f"{stem}{suffix}.mp4"
    out_json = run_dir / f"{stem}{suffix}_detections.json"

    def report(i, n, msg):
        progress(i / max(n, 1), desc=msg)

    s = proc.process(video_path, out_video, out_json, report)
    if puck_focus:
        return (str(out_video) if out_video else None), _puck_summary(s), str(out_json)
    teams = ", ".join(t["color_name"] for t in s["teams"]) or "n/a"
    n = max(s["frames_processed"], 1)
    video_s = n / s["fps"]
    md = (f"**Processed {video_s / 60:.1f} min of video in {s['processing_seconds'] / 60:.1f} min** "
          f"({s['processing_fps']:.1f} frames/s on {' + '.join(s['detector_devices'])})\n\n"
          f"| | |\n|---|---|\n"
          f"| Teams (jersey colors) | {teams} |\n"
          f"| Player tracks | {s['unique_player_tracks']} |\n"
          f"| Puck visible | {100 * s['frames_with_puck'] / n:.0f}% of frames |\n"
          f"| Goal visible | {100 * s['frames_with_goal'] / n:.0f}% of frames |\n"
          f"| Blue line visible | {100 * s['frames_with_blue_line'] / n:.0f}% of frames |\n"
          f"| Center line visible | {100 * s['frames_with_center_line'] / n:.0f}% of frames |\n"
          f"| Detector ran on | {s['keyframes']} keyframes; {s['duplicate_frames']} repeated and "
          f"{s['non_play_frames']} no-rink frames skipped |\n"
          f"| Scene cuts | {s['scene_cuts']} |\n")
    return (str(out_video) if out_video else None), md, str(out_json)


def _puck_summary(s: dict) -> str:
    n = max(s["frames_processed"], 1)
    return (f"**Processed {n / s['fps'] / 60:.1f} min of video in {s['processing_seconds'] / 60:.1f} min** "
            f"({s['processing_fps']:.1f} frames/s on {' + '.join(s['detector_devices'])})\n\n"
            f"| | |\n|---|---|\n"
            f"| Puck located | {100 * s['frames_with_puck'] / n:.0f}% of frames "
            f"({100 * s['frames_with_puck_detected'] / n:.0f}% detected, "
            f"{100 * s['frames_with_puck_interpolated'] / n:.0f}% filled in between detections) |\n"
            f"| Glass-clip false alarms removed | {s['rejected_glass_clips']} |\n"
            f"| Zoomed re-checks | {s.get('zoom_rechecks', 0)} frames, {s.get('zoom_candidates', 0)} extra detections |\n"
            f"| Goal visible | {100 * s['frames_with_goal'] / n:.0f}% of frames |\n"
            f"| Blue line visible | {100 * s['frames_with_blue_line'] / n:.0f}% of frames |\n"
            f"| Center line visible | {100 * s['frames_with_center_line'] / n:.0f}% of frames |\n")


with gr.Blocks(title="Hockey Tracker") as demo:
    gr.Markdown("# Hockey Tracker\nDetects and tracks **players** (split into teams), **referees**, "
                "the **puck**, the **goal**, and the **blue / center lines** in broadcast hockey video.")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Video(label="Input video (.mp4)", sources=["upload"])
            mode = gr.Radio(MODES, value=MODES[0], label="Mode")
            every = gr.Slider(1, 6, value=3, step=1, label="Run the detector on every Nth frame",
                              info="Full analysis only; frames in between are interpolated. Higher is faster. "
                                   "Puck focus always checks every frame.")
            fast = gr.Checkbox(label="Fast mode: lower detector resolution (about 2x faster, finds fewer pucks)", value=False)
            data_only = gr.Checkbox(label="Data only: skip the annotated video (fastest, good for whole games)",
                                    value=False)
            max_s = gr.Number(label="Only process the first N seconds (0 = whole video)", value=0, minimum=0)
            btn = gr.Button("Analyze", variant="primary")
            if SAMPLES:
                gr.Examples([[str(p)] for p in SAMPLES], inputs=[inp], label="Sample videos")
        with gr.Column(scale=2):
            out = gr.Video(label="Annotated video")
            summary = gr.Markdown()
            dl = gr.File(label="Per-frame detections (JSON)")
    btn.click(analyze, inputs=[inp, mode, every, fast, data_only, max_s], outputs=[out, summary, dl])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None,
                    help="port to serve on (default: first free port from 7860 up)")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab on start")
    args = ap.parse_args()
    if not DEFAULT_WEIGHTS.exists():
        raise SystemExit(f"Model weights not found at {DEFAULT_WEIGHTS}. Run: python scripts/download_models.py")
    demo.queue().launch(server_port=args.port, inbrowser=not args.no_browser,
                        allowed_paths=[str(ROOT / "outputs"), str(ROOT / "videos")])
