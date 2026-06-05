"""
orchestrator.py
---------------
Entry point for the Multi-Camera Person ReID system.

Responsibilities:
  - Parse CLI arguments and load config.yaml
  - Set up logging (file + console)
  - Pre-load shared resources (ReidExtractor, GlobalGallery, SpatioTemporalModel)
  - Spawn one CameraPipeline thread per configured camera
  - Wait for all threads to finish (Phase 1) or handle Ctrl+C (Phase 2)
  - Save gallery on exit

Usage:
    python orchestrator.py --config config.yaml

    # Preload models before processing (avoids first-frame freeze):
    python orchestrator.py --config config.yaml --preload

    # Override mode from CLI:
    python orchestrator.py --config config.yaml --mode offline
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import yaml


def setup_logging(log_path: str, log_level: str = "INFO") -> None:
    """Configure root logger with console + rotating file handler."""
    import logging.handlers

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler (rotating: 5MB × 3 backups)
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    fh.setLevel(logging.DEBUG)   # File always gets DEBUG
    fh.setFormatter(fmt)
    root.addHandler(fh)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preload_models(config: dict) -> None:
    """
    Download and cache all models before processing starts.
    Avoids first-frame latency spikes.
    """
    logger = logging.getLogger("preload")
    logger.info("=== Pre-loading models ===")

    # YOLO
    try:
        from ultralytics import YOLO
        yolo_name = config["detection"]["yolo_model"]
        logger.info(f"Loading YOLO: {yolo_name}")
        YOLO(yolo_name)   # This triggers download if needed
        logger.info("YOLO ready ✓")
    except Exception as e:
        logger.error(f"YOLO load failed: {e}")

    # ReID / OSNet
    try:
        from reid_extractor import ReidExtractor
        logger.info("Loading ReID extractor (OSNet)…")
        extractor = ReidExtractor(config["reid"])
        logger.info("ReID extractor ready ✓")
        return extractor
    except Exception as e:
        logger.error(f"ReID extractor load failed: {e}")

    return None


def build_shared_resources(config: dict):
    """Instantiate shared objects used across all camera threads."""
    logger = logging.getLogger("orchestrator")

    logger.info("Building ReID extractor…")
    from reid_extractor import ReidExtractor
    extractor = ReidExtractor(config["reid"])

    logger.info("Building global gallery…")
    from gallery import GlobalGallery
    gallery = GlobalGallery(config)

    logger.info("Building spatio-temporal model…")
    from st_model import SpatioTemporalModel
    st_model = SpatioTemporalModel(config)
    logger.info(st_model.describe())

    return extractor, gallery, st_model


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Camera Person Re-Identification System"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--preload", action="store_true",
        help="Pre-download all models before processing starts"
    )
    parser.add_argument(
        "--mode", type=str, choices=["offline", "live"], default=None,
        help="Override mode from config (offline=video files, live=RTSP)"
    )
    parser.add_argument(
        "--load-gallery", action="store_true",
        help="Load previously saved gallery from disk before starting"
    )
    args = parser.parse_args()

    # ---- Load config ----
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    # CLI override of mode
    if args.mode:
        config["runtime"]["mode"] = args.mode

    # ---- Setup logging ----
    out_cfg  = config.get("output", {})
    log_path  = out_cfg.get("log_path", "logs/reid_runtime.log")
    log_level = out_cfg.get("log_level", "INFO")
    setup_logging(log_path, log_level)

    logger = logging.getLogger("orchestrator")
    logger.info("=" * 60)
    logger.info("Multi-Camera Person ReID System — Starting")
    logger.info(f"Config: {config_path.resolve()}")
    logger.info(f"Mode: {config['runtime']['mode'].upper()}")
    logger.info(f"Cameras: {[c['id'] for c in config['cameras']]}")
    logger.info("=" * 60)

    # ---- Pre-load (optional) ----
    if args.preload:
        preload_models(config)

    # ---- Build shared resources ----
    try:
        extractor, gallery, st_model = build_shared_resources(config)
    except Exception as e:
        logger.exception(f"Failed to initialise shared resources: {e}")
        sys.exit(1)

    # ---- Optionally restore gallery ----
    if args.load_gallery:
        loaded = gallery.load()
        if loaded:
            logger.info(f"Restored gallery: {gallery.size()} entries")
        else:
            logger.warning("No saved gallery found — starting fresh")

    # ---- Shared stop event ----
    stop_event = threading.Event()

    # ---- Spawn camera threads ----
    from camera_pipeline import CameraPipeline

    threads = []
    for cam_cfg in config["cameras"]:
        pipeline = CameraPipeline(
            camera_cfg=cam_cfg,
            full_config=config,
            extractor=extractor,
            gallery=gallery,
            st_model=st_model,
            stop_event=stop_event,
        )
        t = threading.Thread(
            target=pipeline.run,
            name=f"cam-{cam_cfg['id']}",
            daemon=True,
        )
        threads.append(t)

    logger.info(f"Spawning {len(threads)} camera thread(s)…")
    for t in threads:
        t.start()

    # ---- Wait / monitor ----
    mode = config["runtime"]["mode"]
    gallery_autosave = int(config["runtime"].get("gallery_autosave_sec", 300))
    last_save = time.time()

    try:
        while True:
            # Check if all threads are done (Phase 1 — offline)
            alive = [t for t in threads if t.is_alive()]
            if not alive:
                logger.info("All camera threads finished.")
                break

            # Gallery auto-save (useful for Phase 2)
            if mode == "live" and (time.time() - last_save) >= gallery_autosave:
                gallery.save(suffix="_autosave")
                last_save = time.time()

            # Prune stale gallery entries (only in live mode)
            # In offline mode, video timestamps are relative (0–N seconds)
            # while time.time() is wall-clock (~billions), so pruning would
            # instantly kill every entry. Skip it.
            if mode == "live":
                gallery.prune(time.time())

            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("Ctrl+C received — requesting graceful shutdown…")
        stop_event.set()
        for t in threads:
            t.join(timeout=10)

    # ---- Final save ----
    if config["gallery"].get("save_on_exit", True):
        gallery.save(suffix="_final")

    # ---- Print summary ----
    stats = gallery.stats()
    logger.info("=" * 60)
    logger.info("=== SESSION SUMMARY ===")
    logger.info(f"  Total unique global IDs assigned : {stats['total_ids']}")
    logger.info(f"  Cameras active                   : {stats['cameras_active']}")
    logger.info(f"  Gallery persisted to             : {config['gallery']['persistence_path']}")
    logger.info("=" * 60)

    # Print a quick ReID table to console
    print("\n" + "=" * 50)
    print("  FINAL GLOBAL ID REGISTRY")
    print("=" * 50)
    from gallery import GlobalGallery as _G  # just to access gallery directly
    for gid in gallery.all_ids():
        entry = gallery.get_entry(gid)
        cams  = ", ".join(sorted(entry.get("cameras", {})))
        n_obs = entry.get("n_obs", 0)
        t0    = entry.get("first_seen", 0)
        t1    = entry.get("last_seen", 0)
        print(
            f"  {gid}  |  cameras: {cams:<20}  |  "
            f"obs: {n_obs:>4}  |  "
            f"first={t0:.1f}s  last={t1:.1f}s"
        )
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
