# Multi-Camera Person Re-Identification System

Zero-shot, open-set person tracking across multiple CCTV feeds using OSNet + YOLOv8 + ByteTrack.

## Quick Start (Phase 1 — Offline Video Clips)

```bash
# 1. Install main environment
uv sync

# 2. Install torchreid (ReID backbone — vendored due to broken PyPI packaging)
git clone --depth 1 https://github.com/KaiyangZhou/deep-person-reid.git vendor/torchreid
uv pip install tensorboard gdown --no-build-isolation
uv pip install -e vendor/torchreid --no-build-isolation

# 3. Drop your CCTV clips
cp your_cctv1.mp4 data/clips/cctv1.mp4
cp your_cctv2.mp4 data/clips/cctv2.mp4

# 4. Run
uv run reid --config config.yaml
```

Annotated output videos are written to `outputs/annotated/`.

## Phase 2 — Live RTSP Streams

Edit `config.yaml`:
```yaml
runtime:
  mode: live
cameras:
  - id: "CCTV-1"
    source: "rtsp://user:pass@192.168.1.x:554/stream1"
  - id: "CCTV-2"
    source: "rtsp://user:pass@192.168.1.y:554/stream2"
```

Then: `uv run reid --config config.yaml`

## GPU Setup

```bash
uv sync --extra gpu   # installs faiss-gpu instead of faiss-cpu
```

See `project.md` for full architecture documentation.
