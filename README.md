# Hockey Tracker

Takes an MP4 of hockey broadcast footage and identifies, frame by frame:

- **Players on the ice**: tracked with persistent IDs (`ID 7` is a track number, not a jersey number) and split into two teams by jersey color. Goalies are marked `G`.
- **Referees**: labeled separately and excluded from team counts.
- **The puck**: with a short motion trail. A hollow circle means the position is predicted through a brief occlusion.
- **The goal (net)**
- **Blue lines and the center red line**: drawn boards to boards, even when players cover part of the line.

Bench players, coaches, spectators and camera operators are ignored. Only people on the ice are reported.

Output: an annotated H.264 MP4 plus a JSON file with every detection per frame.

There are two modes:

- **Full analysis** (default): everything above.
- **Puck focus** (`--puck-only`, or "Puck focus" in the web UI): only the puck, the lines and the goals, with no players. It works harder to follow the puck precisely: the detector runs on every frame, false pucks are filtered out, and the whole clip is solved at once for the most consistent puck path. It is slower but much more reliable. See [Puck focus](#puck-focus) below.

## Setup (Windows)

```bash
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\download_models.py
```

`download_models.py` fetches the detector and exports an ONNX copy (`models/HockeyAI.onnx`), which lets it run on the GPU.

## Usage

Put your own hockey MP4s in a `videos/` folder. The sample broadcast footage used during development isn't included in this repository.

**Web UI:** upload a video, watch progress, then view or download the results:

```bash
.venv\Scripts\python app.py
```

**Command line:**

```bash
.venv\Scripts\python -m hockey_tracker videos\video1.mp4
.venv\Scripts\python -m hockey_tracker videos\*.mp4 --out outputs
.venv\Scripts\python -m hockey_tracker videos\SPINGOAL.mp4 --start 20 --max-frames 150 --fast
.venv\Scripts\python -m hockey_tracker full_game.mp4 --no-video      # whole game, data only
.venv\Scripts\python -m hockey_tracker videos\video1.mp4 --puck-only  # puck, lines and goals only
```

Each run writes `outputs/<name>_annotated.mp4` and `outputs/<name>_detections.json` (`<name>_puck.mp4` and `<name>_puck_detections.json` in puck focus).

| Option | Effect |
|---|---|
| `--puck-only` | Puck focus mode (see below). |
| `--detect-every N` | Run the detector on every Nth new frame and interpolate the frames in between (default 3, or 1 in puck focus). `1` is the slowest and finds the puck most often. |
| `--device` | `auto` (default) picks the best hardware. Or give a list: `dml,cpu` runs the GPU and the CPU side by side, `cuda` uses an NVIDIA GPU, `cpu` uses the CPU only. |
| `--no-video` | Skip drawing and encoding the annotated video; write only the JSON. |
| `--fast` | Detector at 960 px instead of 1280: about 2x faster, finds fewer pucks. |

**Speed** on a Ryzen 5 7535U laptop, using its Radeon graphics and CPU together (`--device auto`), for a 75-second 1080p screen recording:

| Mode | Time | Frames/s |
|---|---|---|
| Full analysis (default, detector every 3rd frame) | 6 min | ~6 |
| Puck focus (detector every frame, zoomed re-checks) | 28 min | ~1.4 |
| Puck focus with `--detect-every 2` | about half of the above, puck found in ~20% fewer frames | ~3 |

An NVIDIA GPU (`--device cuda`) makes every mode many times faster.

## How it works

| Stage | Method | File |
|---|---|---|
| Color normalization | Each frame is scaled so the ice becomes a bright neutral white, which undoes arena color casts and dim exposure before detection and jersey analysis | `hockey_tracker/color.py` |
| Detection | [HockeyAI](https://huggingface.co/SimulaMet-HOST/HockeyAI) YOLOv8m (MIT), fine-tuned on broadcast hockey: player, goalie, referee, puck, goal, center ice. Runs through ONNX Runtime (DirectML GPU and/or CPU) or PyTorch (CUDA), in background worker threads | `hockey_tracker/detector.py` |
| Keyframes | The detector runs on every Nth new frame only. Repeated frames and no-rink shots (crowd, intermission) are skipped. Frames in between are interpolated between keyframes and corrected for camera motion | `hockey_tracker/pipeline.py` |
| Camera motion | Sparse optical flow → frame-to-frame homography; also detects scene cuts and repeated frames. Dense optical flow measures how each player moved between keyframes | `hockey_tracker/camera.py` |
| Player tracking | BoT-SORT/ByteTrack association, with each track moved by its measured optical flow (IDs survive fast pans and sparse keyframes) | `hockey_tracker/tracking.py` |
| Teams | Jersey Lab-color stats on white-balanced torsos, k-means into 2 teams, per-track voting over time | `hockey_tracker/teams.py` |
| Rink lines | Thick straight stripes darker than the ice (12" paint vs 2" circles/goal lines), classified blue vs red *relative to the ice color*; lines that would cross each other are rejected; the center-ice circle confirms the center line | `hockey_tracker/lines.py` |
| Puck / goal | Detector output filtered to the ice surface, confirmed over several frames, gaps bridged with constant-velocity prediction | `hockey_tracker/tracking.py` |
| Puck focus | Glass-clip rejection, a whole-clip puck path solver and zoomed re-checks (see [Puck focus](#puck-focus)) | `hockey_tracker/puck_focus.py` |

## Puck focus

The detector's raw puck output is noisy. On broadcast footage its most confident "pucks" are often the **clips on top of the glass stanchions** along the near boards, which look exactly like a puck from above. Dark spots in board ads and cameras mounted on the glass also fool it. Puck focus deals with this in three passes (`hockey_tracker/puck_focus.py`):

1. **Analyze.** The detector runs on every new frame at a low confidence threshold, so a faint puck is not missed. A candidate is thrown out if it is off the ice, or if a thin post runs down from it (a glass clip). Rejected clip positions are remembered in rink coordinates, which also removes the same clip in frames where its post is too faint to see. Camera motion is chained into per-shot coordinates, so camera pans don't look like puck motion.
2. **Solve.** For each camera shot, all candidates are solved together as a longest-path problem. It picks at most one candidate per frame, maximizing detector confidence, and only allows steps the puck could physically travel. Anything that sits perfectly still in rink coordinates for a second is treated as fixed equipment. Gaps under 0.4 s are filled with a straight line (drawn as a hollow circle, `"state": "interpolated"` in the JSON). Frames near the path where no puck was found get a **zoomed second look**: a half-frame crop around the expected position is re-detected at 2x magnification, and then the path is solved again.
3. **Render.** A large, glowing puck marker with a trail, plus the lines and goals.

When the puck is buried in a scrum or hidden behind players, puck focus shows nothing rather than guessing.

## Tests

The line detector, team classifier and trackers are tested on synthetic images, so no model weights are needed:

```bash
.venv\Scripts\python -m pip install pytest
.venv\Scripts\python -m pytest tests -q
```

## JSON format

```jsonc
{
  "summary": { "teams": [{"team": 0, "color_name": "white"}, ...], "frames_with_puck": 312, ... },
  "frames": [{
    "frame": 120, "time": 4.0, "keyframe": true, "play": true,
    "players": [{"id": 7, "cls": "player", "team": 1, "box": [x1, y1, x2, y2], "score": 0.88}],
    "puck":    {"center": [x, y], "score": 0.51, "predicted": false},
    "goals":   [{"box": [x1, y1, x2, y2], "score": 0.86}],
    "lines":   [{"kind": "blue_line", "p1": [x, y], "p2": [x, y]}]
  }]
}
```

In puck focus the frame records have no `players`, and the puck is `{"center": [x, y], "state": "detected" | "interpolated", "score": 0.51}`.

Coordinates are pixels in the original video frame. `keyframe` marks frames where the detector ran (the others are interpolated); `play: false` marks frames skipped because no rink was in view.

## Known limitations

- **Puck**: tiny, often motion-blurred or hidden. Expect it to be found in a good share of frames, not all of them. Fast shots are the hardest. Use puck focus when the puck matters most.
- **Puck focus**: a puck lying still for over a second, such as during a stoppage or pinned against the boards, is treated like fixed equipment and not shown. An isolated dark dot in a board ad can occasionally be shown for a moment.
- **Goal**: very reliable in well-lit arenas (video1: about 80% of frames). In dim, tinted arenas like the BU rink in `SPINGOAL.mp4` the net is detected much less often, because the detector was trained on Swedish pro broadcasts.
- **Crowds at the net**: when several players overlap, a track ID can occasionally jump from one person to another.
- **Close-ups and replays**: team colors, lines and the puck are less reliable on tight celebration or bench shots. Tracker state is reset at detected scene cuts.
- **Line color**: on very washed-out footage, blue and red paint differ by only a few color units. The center-ice circle cue and time averaging help, but a single mislabeled line is possible.
- Only the blue lines, center line and goal are drawn. Goal lines and faceoff circles are intentionally filtered out.
