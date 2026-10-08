"""Fast checks of the non-neural components on synthetic data (no model weights needed).

    .venv\\Scripts\\python -m pytest tests -q
"""
import cv2
import numpy as np

from hockey_tracker.lines import RinkLineDetector, drop_crossing, segments_cross
from hockey_tracker.teams import TeamClassifier
from hockey_tracker.tracking import LineTracker, PuckTracker

W, H = 1920, 1080


def _rink(lines):
    """Bluish-white ice with a textured 'boards/stands' band on top and painted stripes."""
    img = np.full((H, W, 3), (235, 228, 222), np.uint8)
    rng = np.random.default_rng(0)
    img[:200] = rng.integers(20, 120, (200, W, 3), dtype=np.uint8)
    img = cv2.add(img, rng.integers(0, 6, img.shape, dtype=np.uint8))
    for (p1, p2, bgr, th) in lines:
        cv2.line(img, p1, p2, bgr, th, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_detects_blue_and_center_lines_and_ignores_thin_circle():
    img = _rink([((500, 250), (200, 1079), (190, 150, 140), 28),     # faded blue line
                 ((1300, 250), (1500, 1079), (170, 165, 215), 28)])  # faded red center line
    cv2.circle(img, (960, 650), 250, (170, 165, 215), 5)              # 2" faceoff circle: not a line
    lines = RinkLineDetector().detect(img)
    kinds = sorted(l.kind for l in lines)
    assert kinds == ["blue_line", "center_line"], [(l.kind, l.delta_lab) for l in lines]


def test_person_boxes_do_not_create_lines():
    img = _rink([])
    cv2.rectangle(img, (900, 400), (960, 700), (40, 40, 200), -1)   # a red 'player'
    assert RinkLineDetector().detect(img, np.array([[890, 390, 970, 710]], np.float32)) == []


def test_segments_cross_and_drop_crossing():
    assert segments_cross((0, 0), (10, 10), (0, 10), (10, 0))
    assert not segments_cross((0, 0), (10, 0), (0, 5), (10, 5))

    class L:
        def __init__(self, p1, p2, s):
            self.p1, self.p2, self.s = np.array(p1, float), np.array(p2, float), s
    kept = drop_crossing([L((0, 0), (10, 10), 1), L((0, 10), (10, 0), 5), L((20, 0), (20, 10), 2)],
                         key=lambda l: l.s)
    assert sorted(k.s for k in kept) == [2, 5]


def test_team_classifier_separates_red_and_white():
    rng = np.random.default_rng(1)
    frame = np.full((600, 1200, 3), (230, 230, 230), np.uint8)
    boxes = []
    for i in range(12):
        x = 40 + i * 95
        color = (40, 40, 200) if i % 2 else (245, 245, 245)
        cv2.rectangle(frame, (x, 100), (x + 60, 260), [int(c + rng.integers(-15, 15)) for c in color], -1)
        cv2.rectangle(frame, (x + 10, 260), (x + 50, 400), (30, 30, 30), -1)   # pants
        boxes.append([x, 80, x + 60, 400])
    tc = TeamClassifier()
    tc.add_samples(frame, np.array(boxes, np.float32))
    assert tc.fit()
    pred = tc.predict(frame, boxes)
    assert len(set(pred[0::2])) == 1 and len(set(pred[1::2])) == 1 and pred[0] != pred[1]
    assert sorted(tc.team_names) == ["red", "white"]


def test_puck_tracker_confirms_then_predicts_through_gap():
    pt = PuckTracker(frame_diag=float(np.hypot(W, H)))
    I = np.eye(3)
    box = lambda x: (np.array([x, 500, x + 12, 508], np.float32), 0.3)
    assert pt.update([box(100)], I) is None                  # one weak detection: not yet
    s = pt.update([box(120)], I)
    assert s is not None and not s["predicted"]               # consistent second hit: confirmed
    s = pt.update([], I)
    assert s is not None and s["predicted"] and s["center"][0] > 126   # keeps moving through a gap
    for _ in range(10):
        s = pt.update([], I)
    assert s is None                                          # eventually dropped


def test_puck_tracker_follows_camera_pan():
    pt = PuckTracker(frame_diag=float(np.hypot(W, H)))
    pan = np.array([[1, 0, -50], [0, 1, 0], [0, 0, 1]], float)   # camera pans right: scene moves left 50px
    b = lambda x: (np.array([x, 500, x + 12, 508], np.float32), 0.4)
    pt.update([b(600)], np.eye(3))
    pt.update([b(600)], np.eye(3))
    s = pt.update([b(552)], pan)                               # puck static on ice, moved only by the pan
    assert s is not None and not s["predicted"] and abs(s["center"][0] - 558) < 1


def test_interpolation_between_keyframes_follows_camera_motion():
    from hockey_tracker.pipeline import blend_states, warp_state
    pan = np.array([[1, 0, -30], [0, 1, 0], [0, 0, 1]], float)       # camera pans: scene shifts 30 px left
    player = lambda x, pid=1: {"id": pid, "box": np.array([x, 100, x + 40, 200], np.float32), "cls": "player",
                                 "team": 0, "score": 0.9}
    a = {"players": [player(100), player(500, 2)], "puck": None, "goals": [], "lines": []}   # keyframe k0
    b = {"players": [player(160)], "puck": None, "goals": [], "lines": []}                   # keyframe k1
    # halfway frame: k0 state moved by half the pan, k1 state moved back by the other half
    half = np.array([[1, 0, -15], [0, 1, 0], [0, 0, 1]], float)
    mid = blend_states(warp_state(a, half), warp_state(b, half @ np.linalg.inv(pan)), 0.5)
    p1 = next(p for p in mid["players"] if p["id"] == 1)
    assert abs(p1["box"][0] - (0.5 * 85 + 0.5 * 175)) < 1e-3           # lerp of 100-15 and 160+15
    assert [p["id"] for p in mid["players"]] == [1]                     # player 2 left the view at k1
    early = blend_states(warp_state(a, half), warp_state(b, half @ np.linalg.inv(pan)), 0.2)
    assert sorted(p["id"] for p in early["players"]) == [1, 2]


def test_glass_post_score_separates_clips_from_pucks():
    from hockey_tracker.puck_focus import CLIP_SCORE, glass_post_score
    gray = np.full((400, 400), 225, np.uint8)
    cv2.circle(gray, (100, 100), 5, 30, -1)                       # clip: dark dot...
    cv2.line(gray, (100, 110), (108, 300), 214, 2)                 # ...on a faint, slightly slanted post
    cv2.circle(gray, (300, 100), 5, 30, -1)                        # puck: dark dot on open ice
    assert glass_post_score(gray, 100, 100, 12) >= CLIP_SCORE
    assert glass_post_score(gray, 300, 100, 12) < 0.1


def test_puck_path_follows_moving_puck_and_skips_equipment():
    from hockey_tracker.puck_focus import path_positions, solve_path
    rng = np.random.default_rng(3)
    diag, fps_u, n = 2200.0, 15.0, 60
    cands = {}
    for t in range(n):
        rows = []
        if t % 7 != 3:                                             # the puck, missed now and then
            rows.append((400 + 12 * t, 500 + 3 * t, 12, 0.5, 0.0, 0, 0))
        rows.append((1500, 300, 12, 0.6, 0.0, 0, 0))                     # bright but motionless: equipment
        if rng.random() < 0.3:                                     # scattered one-off false alarms
            rows.append((rng.uniform(0, 1900), rng.uniform(0, 1000), 12, 0.2, 0.0, 0, 0))
        cands[t] = np.array(rows, np.float64)
    pos = path_positions(solve_path(cands, n, diag, fps_u))
    assert len(pos) >= 0.9 * n
    for t, (p, size, score, kind) in pos.items():
        assert np.linalg.norm(p - (400 + 12 * t, 500 + 3 * t)) < 1.0, (t, p, kind)
    assert pos[3][3] == "interpolated" and pos[4][3] == "detected"


def test_overlay_check_flags_screen_fixed_patches_only():
    from hockey_tracker.puck_focus import OverlayCheck
    rng = np.random.default_rng(0)
    scene = rng.integers(60, 200, (600, 1400), dtype=np.uint8)          # textured "rink"
    cursor = rng.integers(0, 255, (24, 24), dtype=np.uint8)
    oc = OverlayCheck()
    for k in range(8):                                                  # camera pans 10 px per frame
        frame = scene[:, 10 * k:10 * k + 1000].copy()
        frame[200:224, 300:324] = cursor                                # overlay: same pixels every frame
        M = np.array([[1, 0, 10 * k], [0, 1, 0], [0, 0, 1]], float)     # frame -> shot coordinates
        if k == 7:
            assert oc.is_overlay(frame, M, 0, 312, 212)                 # the cursor
            assert not oc.is_overlay(frame, M, 0, 600, 400)             # scene content moving with the pan
        oc.push(frame, M, 0)


def test_drop_clip_spots_removes_known_clip_locations():
    from hockey_tracker.puck_focus import drop_clip_spots
    cands = {0: np.array([[100.0, 100.0, 12, 0.5, 0.0], [800.0, 400.0, 12, 0.5, 0.0]]),
             1: np.array([[102.0, 99.0, 12, 0.5, 0.0]])}
    out = drop_clip_spots(cands, [(101.0, 100.0)], 2200.0)
    assert out[0][:, :2].tolist() == [[800.0, 400.0]] and len(out[1]) == 0


def test_line_tracker_needs_consistency_and_keeps_one_center_line():
    from hockey_tracker.lines import RinkLine
    mk = lambda x, red: RinkLine("center_line" if red else "blue_line", np.array([x, 200.0]), np.array([x - 150, 1000.0]),
                                 20.0, (-25.0, 3.0 if red else 0.5, 1.0 if red else -2.0), 300.0, 1.0)
    lt = LineTracker(W)
    I = np.eye(3)
    assert lt.update([mk(500, False), mk(1400, True)], I) == []     # needs a few frames
    lt.update([mk(500, False), mk(1400, True)], I)
    out = lt.update([mk(500, False), mk(1400, True)], I)
    assert sorted(l["kind"] for l in out) == ["blue_line", "center_line"]
