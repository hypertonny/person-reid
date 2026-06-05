"""
colab_setup.py
--------------
Google Colab bootstrap script for Multi-Camera Person ReID.

Usage (in a Colab cell):
    !git clone <your-repo-url> /content/person-reid
    %cd /content/person-reid/multi_camera_reid
    !python colab_setup.py

Or paste the install steps below directly into Colab cells.
"""

import subprocess
import sys
import os


def run(cmd):
    """Run a shell command and stream output."""
    print(f"\n{'='*60}")
    print(f">>> {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"⚠️  Command exited with code {result.returncode}")
    return result.returncode


def main():
    # ---- 1. Core dependencies ----
    run("pip install -q ultralytics torch torchvision faiss-gpu scipy opencv-python pyyaml numpy")

    # ---- 2. torchreid (from git, needs build deps) ----
    run("pip install -q Cython gdown tensorboard")
    run("pip install -q git+https://github.com/KaiyangZhou/deep-person-reid.git")

    # ---- 3. ByteTrack dependency ----
    run("pip install -q lap")

    # ---- 4. Create directory structure ----
    dirs = [
        "data/clips",
        "data/test_videos",
        "models",
        "outputs/annotated",
        "outputs/gallery_state",
        "logs",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  ✅ {d}/")

    # ---- 5. Verify imports ----
    print(f"\n{'='*60}")
    print("Verifying imports...")
    print('='*60)
    try:
        import torch
        print(f"  ✅ torch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  ❌ torch")

    try:
        import torchreid
        print(f"  ✅ torchreid {torchreid.__version__}")
    except ImportError:
        print("  ❌ torchreid")

    try:
        from ultralytics import YOLO
        print(f"  ✅ ultralytics")
    except ImportError:
        print("  ❌ ultralytics")

    try:
        import faiss
        print(f"  ✅ faiss (GPU: {hasattr(faiss, 'GpuIndexFlatIP')})")
    except ImportError:
        print("  ❌ faiss")

    try:
        import scipy
        print(f"  ✅ scipy {scipy.__version__}")
    except ImportError:
        print("  ❌ scipy")

    try:
        import cv2
        print(f"  ✅ opencv {cv2.__version__}")
    except ImportError:
        print("  ❌ opencv")

    print(f"\n{'='*60}")
    print("🚀 Setup complete! Upload your clips and run:")
    print("   python orchestrator.py --config config.yaml")
    print('='*60)


if __name__ == "__main__":
    main()
