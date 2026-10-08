"""Command line entry point.

    python -m hockey_tracker videos/video1.mp4
    python -m hockey_tracker game.mp4 --no-video                 # whole game, data only
    python -m hockey_tracker videos/*.mp4 --out outputs --detect-every 1   # max accuracy
    python -m hockey_tracker videos/video1.mp4 --puck-only       # puck, lines and goals only
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from .detector import available_devices
from .pipeline import DEFAULT_WEIGHTS, Settings, VideoProcessor
from .puck_focus import PuckFocusProcessor


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hockey_tracker",
                                 description="Detect players, puck, goals and rink lines in hockey video.")
    ap.add_argument("videos", nargs="+", help="input .mp4 file(s); globs are expanded")
    ap.add_argument("--out", default="outputs", help="output directory (default: outputs)")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="HockeyAI detector weights (.pt)")
    ap.add_argument("--device", default="auto",
                    help=f"detector hardware: auto, or a comma list of {'/'.join(available_devices())} "
                         "(e.g. 'dml,cpu' runs the GPU and the CPU in parallel)")
    ap.add_argument("--puck-only", action="store_true",
                    help="puck-focus mode: track the puck as precisely as possible plus rink lines and goals, "
                         "no players (detector on every frame by default)")
    ap.add_argument("--detect-every", type=int, default=None,
                    help="run the detector on every Nth new frame and interpolate in between "
                         "(default 3, or 1 with --puck-only; 1 = every frame, slowest and most accurate)")
    ap.add_argument("--fast", action="store_true", help="run the detector at 960 px (~2x faster, fewer pucks)")
    ap.add_argument("--no-video", action="store_true", help="skip the annotated video, write only the JSON")
    ap.add_argument("--max-frames", type=int, default=None, help="only process this many frames")
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    args = ap.parse_args(argv)

    paths = [Path(p) for pat in args.videos for p in (glob.glob(pat) or [pat])]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"Not found: {', '.join(map(str, missing))}", file=sys.stderr)
        return 2
    if not Path(args.weights).exists():
        print(f"Model weights not found at {args.weights}. Run: python scripts/download_models.py", file=sys.stderr)
        return 2

    every = args.detect_every if args.detect_every is not None else (1 if args.puck_only else 3)
    settings = Settings(weights=Path(args.weights), imgsz=960 if args.fast else 1280, device=args.device,
                        detect_every=every, render_video=not args.no_video,
                        max_frames=args.max_frames, start_s=args.start)
    proc = PuckFocusProcessor(settings) if args.puck_only else VideoProcessor(settings)
    suffix = "_puck" if args.puck_only else "_annotated"
    print(f"Detector workers: {', '.join(proc.pool.names)}")
    out_dir = Path(args.out)
    try:
        for path in paths:
            print(f"\n== {path}")
            last = {"stage": None}

            def progress(i, n, msg):
                stage = msg.split(" ")[0]
                if stage != last["stage"] or i % 100 == 0 or i == n:
                    print(f"  {msg}", flush=True)
                    last["stage"] = stage

            summary = proc.process(path, None if args.no_video else out_dir / f"{path.stem}{suffix}.mp4",
                                   out_dir / f"{path.stem}{suffix}_detections.json", progress)
            print(json.dumps(summary, indent=2))
    finally:
        proc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
