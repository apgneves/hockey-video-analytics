"""Object detection with the HockeyAI YOLOv8 model (players, goalies, referees, puck, goal).

Backends:
  * ONNX Runtime ("dml" = DirectML on any Windows GPU incl. AMD/Intel integrated graphics,
    "cuda", or "cpu"). Uses models/HockeyAI.onnx (see scripts/download_models.py).
  * PyTorch / Ultralytics ("torch-cuda", "torch-cpu") using models/HockeyAI.pt.

DetectorPool runs several backends at once (e.g. the GPU and the CPU) on different frames.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# HockeyAI class names -> our names
PLAYER, GOALIE, REFEREE, PUCK, GOAL = "player", "goalie", "referee", "puck", "goal"
CENTER_ICE = "center_ice"
PERSON_CLASSES = (PLAYER, GOALIE, REFEREE)
CLASS_NAMES = ["center_ice", "faceoff", GOAL, GOALIE, PLAYER, PUCK, REFEREE]   # model's class order


@dataclass
class Detections:
    """Detections for one frame. Boxes are float32 xyxy in full-frame pixels."""
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    labels: list[str] = field(default_factory=list)

    def select(self, *names: str, min_score: float = 0.0) -> "Detections":
        keep = [i for i, n in enumerate(self.labels) if n in names and self.scores[i] >= min_score]
        return Detections(self.boxes[keep], self.scores[keep], [self.labels[i] for i in keep])

    def __len__(self) -> int:
        return len(self.labels)


# --------------------------------------------------------------------------- ONNX Runtime
def _letterbox(img: np.ndarray, size: int, stride: int = 32):
    """Resize keeping aspect ratio and pad to a multiple of `stride` (Ultralytics 'rect' mode)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (size - nw) % stride / 2, (size - nh) % stride / 2
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, left, top


class OnnxBackend:
    def __init__(self, onnx_path: str | Path, imgsz: int, provider: str = "dml", threads: int | None = None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if threads:
            so.intra_op_num_threads = threads
        providers = {"dml": ["DmlExecutionProvider"], "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                     "cpu": ["CPUExecutionProvider"]}[provider]
        if provider == "dml":
            so.enable_mem_pattern = False          # required by DirectML
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if providers[0] not in ort.get_available_providers():
            raise RuntimeError(f"ONNX Runtime provider {providers[0]} is not available")
        self.session = ort.InferenceSession(str(onnx_path), so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.name = f"onnx-{provider}"

    def __call__(self, frame_bgr: np.ndarray, conf: float = 0.08, iou: float = 0.7, max_det: int = 100) -> Detections:
        img, r, left, top = _letterbox(frame_bgr, self.imgsz)
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        pred = self.session.run(None, {self.input_name: x})[0][0].T     # (anchors, 4 + classes)
        scores_all = pred[:, 4:]
        cls = scores_all.argmax(1)
        scores = scores_all[np.arange(len(cls)), cls]
        keep = scores >= conf
        if not keep.any():
            return Detections()
        xywh, scores, cls = pred[keep, :4], scores[keep], cls[keep]
        boxes = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
        idx = cv2.dnn.NMSBoxesBatched(
            [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
            scores.astype(float).tolist(), cls.astype(int).tolist(), conf, iou)
        idx = np.asarray(idx, int).reshape(-1)
        idx = idx[np.argsort(-scores[idx])][:max_det]
        boxes = (boxes[idx] - np.array([left, top, left, top], np.float32)) / r
        h, w = frame_bgr.shape[:2]
        boxes = np.clip(boxes, 0, [w, h, w, h]).astype(np.float32)
        return Detections(boxes, scores[idx].astype(np.float32), [CLASS_NAMES[c] for c in cls[idx]])


# --------------------------------------------------------------------------- PyTorch
class TorchBackend:
    def __init__(self, weights: str | Path, imgsz: int, device: str = "cpu"):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.device = device
        self.precision = {"quantize": 16} if device.startswith("cuda") else {}
        self.name = f"torch-{device}"
        raw = self.model.names
        self._names = {i: {"centriod": CENTER_ICE}.get(n, n) for i, n in raw.items()}

    def __call__(self, frame_bgr: np.ndarray, conf: float = 0.08) -> Detections:
        r = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=conf, device=self.device, verbose=False,
                               max_det=100, **self.precision)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return Detections()
        return Detections(
            boxes=b.xyxy.cpu().numpy().astype(np.float32),
            scores=b.conf.cpu().numpy().astype(np.float32),
            labels=[self._names[int(c)] for c in b.cls.cpu().numpy()],
        )


# --------------------------------------------------------------------------- selection
def available_devices() -> list[str]:
    out = []
    try:
        import torch
        if torch.cuda.is_available():
            out.append("cuda")
    except ImportError:
        pass
    try:
        import onnxruntime as ort
        prov = ort.get_available_providers()
        if "CUDAExecutionProvider" in prov and "cuda" not in out:
            out.append("cuda")
        if "DmlExecutionProvider" in prov:
            out.append("dml")
    except ImportError:
        pass
    out.append("cpu")
    return out


def make_backend(device: str, weights: str | Path, imgsz: int, threads: int | None = None):
    """device: cuda | dml | cpu. Prefers ONNX Runtime when the .onnx model exists."""
    weights = Path(weights)
    onnx = weights.with_suffix(".onnx")
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return TorchBackend(weights, imgsz, "cuda:0")
        except ImportError:
            pass
        return OnnxBackend(onnx, imgsz, "cuda")
    if device == "dml":
        return OnnxBackend(onnx, imgsz, "dml")
    if onnx.exists():
        return OnnxBackend(onnx, imgsz, "cpu", threads)   # ~20% faster than PyTorch on CPU
    return TorchBackend(weights, imgsz, "cpu")


def resolve_devices(spec: str) -> list[str]:
    """'auto' -> best accelerator, plus the CPU as a second worker when the accelerator is an
    integrated GPU (the CPU is fast enough to add meaningful throughput alongside it).
    Otherwise a comma list, e.g. 'dml,cpu' or 'cuda'."""
    avail = available_devices()
    if spec == "auto":
        if "cuda" in avail:
            return ["cuda"]
        if "dml" in avail:
            return ["dml", "cpu"]
        return ["cpu"]
    devs = [d.strip() for d in spec.split(",") if d.strip()]
    missing = [d for d in devs if d not in avail]
    if missing:
        raise RuntimeError(f"Device(s) not available: {missing}. Available: {avail}")
    return devs


class DetectorPool:
    """Runs detection on one or more backends in background threads. submit() returns a
    Future; frames may finish out of order across backends, callers consume in order.
    ONNX Runtime and PyTorch release the GIL while computing, so the backends and the main
    thread (tracking, line finding, rendering) all run concurrently."""

    def __init__(self, devices: list[str], weights: str | Path, imgsz: int, cpu_threads: int | None = None):
        self.jobs: queue.Queue = queue.Queue()
        self.backends = []
        for d in devices:
            # when the CPU shares work with a GPU, leave cores for the main thread
            threads = cpu_threads if (d == "cpu" and len(devices) > 1) else None
            self.backends.append(make_backend(d, weights, imgsz, threads))
        self.threads = [threading.Thread(target=self._work, args=(b,), daemon=True) for b in self.backends]
        for t in self.threads:
            t.start()

    @property
    def names(self) -> list[str]:
        return [b.name for b in self.backends]

    def _work(self, backend) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            frame, fut = job
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(backend(frame))
            except BaseException as e:   # surface errors to the consumer
                fut.set_exception(e)

    def submit(self, frame: np.ndarray) -> Future:
        fut: Future = Future()
        self.jobs.put((frame, fut))
        return fut

    def __call__(self, frame: np.ndarray) -> Detections:
        return self.submit(frame).result()

    def close(self) -> None:
        for _ in self.threads:
            self.jobs.put(None)
