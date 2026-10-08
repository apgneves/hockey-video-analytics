"""Download the pretrained HockeyAI detector (MIT license, SimulaMet-HOST) into models/ and
export an ONNX copy, which runs on GPUs through ONNX Runtime (DirectML on any Windows GPU,
including AMD/Intel integrated graphics) and is ~20% faster than PyTorch on the CPU."""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/SimulaMet-HOST/HockeyAI/resolve/main/HockeyAI_model_weight.pt"
DEST = Path(__file__).resolve().parent.parent / "models" / "HockeyAI.pt"


def download() -> None:
    if DEST.exists():
        print(f"Already present: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}\n  -> {DEST}")
    tmp = DEST.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    tmp.replace(DEST)
    print(f"Done ({DEST.stat().st_size / 1e6:.1f} MB)")


def export_onnx() -> None:
    onnx = DEST.with_suffix(".onnx")
    if onnx.exists():
        print(f"Already present: {onnx}")
        return
    # Ultralytics would otherwise "fix" requirements by installing the CPU-only onnxruntime
    # over onnxruntime-directml, which silently disables the GPU.
    os.environ["YOLO_AUTOINSTALL"] = "False"
    from ultralytics import YOLO
    print(f"Exporting {onnx.name} (dynamic input size) ...")
    out = YOLO(str(DEST)).export(format="onnx", imgsz=1280, dynamic=True, simplify=True, opset=17)
    Path(out).replace(onnx)
    print(f"Done: {onnx}")


if __name__ == "__main__":
    download()
    export_onnx()
    sys.exit(0)
