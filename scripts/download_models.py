"""Download pretrained model files into models/ (gitignored). §6, §21.4."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size)
    print(f"\r  {pct:3d}%", end="", flush=True)


def download_all() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"{filename}: already present, skipping")
            continue
        print(f"{filename}: downloading from {url}")
        try:
            urllib.request.urlretrieve(url, dest, reporthook=_progress)
            print()
        except Exception as exc:  # noqa: BLE001 - report and stop
            print(f"\n  FAILED: {exc}")
            if dest.exists():
                dest.unlink()
            sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    download_all()