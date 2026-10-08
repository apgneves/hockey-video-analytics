"""Draw annotations onto frames."""
from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX
LINE_STYLE = {
    "blue_line": ((235, 120, 20), "BLUE LINE"),
    "center_line": ((40, 40, 230), "CENTER LINE"),
}
REF_COLOR = (60, 60, 60)
PUCK_COLOR = (0, 230, 255)
GOAL_COLOR = (255, 0, 255)
UNKNOWN_TEAM = (160, 160, 160)


def _label(img, text, org, color, scale, thickness=1, bg=True):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = int(org[0]), int(org[1])
    x = min(max(x, 0), img.shape[1] - tw - 4)
    y = max(y, th + 4)
    if bg:
        lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + base - 1), color, -1)
        txt = (0, 0, 0) if lum > 150 else (255, 255, 255)
        cv2.putText(img, text, (x + 2, y - 2), FONT, scale, txt, thickness, cv2.LINE_AA)
    else:
        cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


class Renderer:
    def __init__(self, frame_w: int, frame_h: int, team_colors: list, team_names: list, puck_focus: bool = False):
        """puck_focus: puck-only mode (no player rows in the legend, a larger, glowing puck marker)."""
        self.s = frame_h / 1080.0
        self.team_colors = [tuple(int(v) for v in c) for c in team_colors]
        self.team_names = team_names
        self.puck_focus = puck_focus

    def team_color(self, team: int | None):
        if team is None or team >= len(self.team_colors):
            return UNKNOWN_TEAM
        return self.team_colors[team]

    def draw(self, frame: np.ndarray, players: list[dict], puck: dict | None,
             goals: list[dict], lines: list[dict]) -> np.ndarray:
        img = frame.copy()
        s = self.s
        self._draw_lines(img, lines)

        for g in goals:
            x1, y1, x2, y2 = g["box"].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), GOAL_COLOR, max(2, int(3 * s)), cv2.LINE_AA)
            _label(img, "GOAL", (x1, y1 - 4), GOAL_COLOR, 0.6 * s + 0.1, 1)

        for p in sorted(players, key=lambda p: p["box"][3]):
            x1, y1, x2, y2 = p["box"].astype(int)
            if p["cls"] == "referee":
                color, tag = REF_COLOR, "REF"
            else:
                color = self.team_color(p.get("team"))
                tag = ("G " if p["cls"] == "goalie" else "") + f"ID {p['id']}"
            cx, w = (x1 + x2) // 2, x2 - x1
            axes = (max(8, int(0.55 * w)), max(4, int(0.16 * w)))
            th = max(2, int(3 * s))
            # ellipse at the feet (dark outline keeps white-team markers visible on the ice)
            cv2.ellipse(img, (cx, y2), axes, 0, -45, 235, (20, 20, 20), th + 2, cv2.LINE_AA)
            cv2.ellipse(img, (cx, y2), axes, 0, -45, 235, color, th, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), max(1, int(1.5 * s)) + 2, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, int(1.5 * s)), cv2.LINE_AA)
            _label(img, tag, (x1, y1 - 3), color, 0.5 * s + 0.1, 1)

        if puck is not None:
            self._draw_puck(img, puck)
        self._draw_legend(img, players, puck, goals, lines)
        return img

    def _draw_lines(self, img, lines):
        s = self.s
        overlay = img.copy()
        for ln in lines:
            color, _ = LINE_STYLE[ln["kind"]]
            th = int(np.clip(ln.get("thickness", 20) * 0.6, 6 * s, 26 * s))
            cv2.line(overlay, tuple(ln["p1"].astype(int)), tuple(ln["p2"].astype(int)), color, th, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)
        for ln in lines:
            color, name = LINE_STYLE[ln["kind"]]
            p1, p2 = ln["p1"], ln["p2"]
            cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), color, max(2, int(2 * s)), cv2.LINE_AA)
            top = p1 if p1[1] < p2[1] else p2
            bot = p2 if top is p1 else p1
            anchor = top + 0.12 * (bot - top)
            _label(img, name, (anchor[0] + 8 * s, anchor[1]), color, 0.55 * s + 0.1, 1)

    def _draw_puck(self, img, puck):
        s = self.s
        focus = self.puck_focus
        trail = puck["trail"] + [puck["center"]]
        max_step = (0.08 if focus else 0.05) * float(np.hypot(*img.shape[:2]))
        width = 5 if focus else 3
        for i in range(len(trail) - 1, 0, -1):
            a, b = trail[i - 1], trail[i]
            if np.linalg.norm(b - a) > max_step:   # a re-acquisition jump, not real motion
                break
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)),
                     PUCK_COLOR, max(1, int(width * s * i / len(trail))), cv2.LINE_AA)
        c = tuple(puck["center"].astype(int))
        r = int(max((14 if focus else 10) * s, puck["size"] * 0.9))
        if focus:
            # soft glow so the puck is easy to follow at full speed
            x0, y0 = max(0, c[0] - 3 * r), max(0, c[1] - 3 * r)
            roi = img[y0:c[1] + 3 * r, x0:c[0] + 3 * r]
            if roi.size:
                glow = roi.copy()
                cv2.circle(glow, (c[0] - x0, c[1] - y0), int(2.2 * r), PUCK_COLOR, -1, cv2.LINE_AA)
                cv2.addWeighted(glow, 0.15 if puck["predicted"] else 0.3, roi, 0.85 if puck["predicted"] else 0.7, 0, dst=roi)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):   # crosshair ticks
                p1 = (c[0] + dx * (r + int(4 * s)), c[1] + dy * (r + int(4 * s)))
                p2 = (c[0] + dx * (r + int(12 * s)), c[1] + dy * (r + int(12 * s)))
                cv2.line(img, p1, p2, PUCK_COLOR, max(2, int(3 * s)), cv2.LINE_AA)
        if puck["predicted"]:
            cv2.circle(img, c, r, PUCK_COLOR, max(1, int(2 * s)), cv2.LINE_AA)
        else:
            cv2.circle(img, c, r + int(4 * s), (0, 0, 0), max(2, int(3 * s)), cv2.LINE_AA)
            cv2.circle(img, c, r, PUCK_COLOR, max(2, int(4 * s)), cv2.LINE_AA)
        _label(img, "PUCK", (c[0] + r + int(14 * s if focus else 4), c[1] - r), PUCK_COLOR, 0.5 * s + 0.1, 1)

    def _draw_legend(self, img, players, puck, goals, lines):
        s = self.s
        rows = []
        if not self.puck_focus:
            for t, name in enumerate(self.team_names):
                n = sum(1 for p in players if p.get("team") == t and p["cls"] != "referee")
                rows.append((self.team_colors[t], f"{name.title()}: {n}"))
            nref = sum(1 for p in players if p["cls"] == "referee")
            rows.append((REF_COLOR, f"Referees: {nref}"))
        rows.append((PUCK_COLOR, "Puck: " + ("tracked" if puck and not puck["predicted"] else
                                             "interpolated" if puck and self.puck_focus else
                                             "predicted" if puck else "not found")))
        rows.append((GOAL_COLOR, f"Goals: {len(goals)}"))
        kinds = sorted({ln["kind"] for ln in lines})
        rows.append(((235, 120, 20), "Lines: " + (", ".join(LINE_STYLE[k][1].title() for k in kinds) or "none")))
        scale = 0.5 * s + 0.1
        lh = int(24 * s + 6)
        w = int(260 * s + 20)
        x0, y0 = img.shape[1] - w - int(12 * s), int(12 * s)
        box = img[y0:y0 + lh * len(rows) + 10, x0:x0 + w]
        box[:] = (box * 0.35).astype(np.uint8)
        for i, (color, text) in enumerate(rows):
            y = y0 + 8 + lh * i + lh // 2
            cv2.circle(img, (x0 + 14, y), max(4, int(7 * s)), color, -1, cv2.LINE_AA)
            cv2.putText(img, text, (x0 + 28, y + int(6 * s)), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
