import os
import shutil
import threading
import time
import yaml
import logging
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_server")

app = FastAPI(title="Multi-Camera ReID Web Dashboard")

# Ensure required directories exist
PROJECT_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = PROJECT_DIR / "data" / "clips"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "annotated"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mount outputs/annotated directory to serve the videos directly
app.mount("/outputs/annotated", StaticFiles(directory=str(OUTPUT_DIR)), name="annotated")

# Global pipeline state
PIPELINE_STATUS = {
    "status": "idle",       # idle, running, converting, completed, failed
    "progress": 0.0,
    "current_step": "Ready to process videos",
    "error": None,
    "cctv1_frames": 0,
    "cctv2_frames": 0,
    "cctv1_total": 0,
    "cctv2_total": 0,
}

ACTIVE_PIPELINES = []
stop_event = threading.Event()

def run_reid_pipeline_bg(cctv1_path: str, cctv2_path: str):
    global PIPELINE_STATUS, ACTIVE_PIPELINES, stop_event
    try:
        PIPELINE_STATUS["status"] = "running"
        PIPELINE_STATUS["progress"] = 0.0
        PIPELINE_STATUS["current_step"] = "Loading models and config..."
        PIPELINE_STATUS["error"] = None
        
        # Load config
        config_path = PROJECT_DIR / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        # Update sources in config dynamically for this run
        config["cameras"][0]["source"] = cctv1_path
        config["cameras"][1]["source"] = cctv2_path
        config["runtime"]["mode"] = "offline"
        config["runtime"]["display"] = False
        
        # Build shared resources
        PIPELINE_STATUS["current_step"] = "Initializing ReID Extractor & Gallery..."
        from reid_extractor import ReidExtractor
        from gallery import GlobalGallery
        from st_model import SpatioTemporalModel
        from camera_pipeline import CameraPipeline
        
        extractor = ReidExtractor(config["reid"])
        gallery = GlobalGallery(config)
        st_model = SpatioTemporalModel(config)
        
        stop_event.clear()
        ACTIVE_PIPELINES = []
        
        # Instantiate pipelines
        for cam_cfg in config["cameras"]:
            pipeline = CameraPipeline(
                camera_cfg=cam_cfg,
                full_config=config,
                extractor=extractor,
                gallery=gallery,
                st_model=st_model,
                stop_event=stop_event,
            )
            ACTIVE_PIPELINES.append(pipeline)
            
        # Spawn threads
        threads = []
        for p in ACTIVE_PIPELINES:
            t = threading.Thread(target=p.run, daemon=True)
            threads.append(t)
            
        PIPELINE_STATUS["current_step"] = "Processing video frames (YOLO + OSNet)..."
        for t in threads:
            t.start()
            
        # Monitor progress
        while any(t.is_alive() for t in threads):
            # Calculate progress
            total_processed = sum(p._frame_count for p in ACTIVE_PIPELINES)
            total_frames = sum(max(1, getattr(p, "total_frames", 100)) for p in ACTIVE_PIPELINES)
            
            p1_count = ACTIVE_PIPELINES[0]._frame_count if len(ACTIVE_PIPELINES) > 0 else 0
            p2_count = ACTIVE_PIPELINES[1]._frame_count if len(ACTIVE_PIPELINES) > 1 else 0
            p1_total = getattr(ACTIVE_PIPELINES[0], "total_frames", 0) if len(ACTIVE_PIPELINES) > 0 else 0
            p2_total = getattr(ACTIVE_PIPELINES[1], "total_frames", 0) if len(ACTIVE_PIPELINES) > 1 else 0
            
            PIPELINE_STATUS["cctv1_frames"] = p1_count
            PIPELINE_STATUS["cctv2_frames"] = p2_count
            PIPELINE_STATUS["cctv1_total"] = p1_total
            PIPELINE_STATUS["cctv2_total"] = p2_total
            
            if total_frames > 0:
                PIPELINE_STATUS["progress"] = round((total_processed / total_frames) * 100, 1)
                
            PIPELINE_STATUS["current_step"] = f"Processing frames: CCTV-1 ({p1_count}/{p1_total}) | CCTV-2 ({p2_count}/{p2_total})"
            time.sleep(0.5)
            
        # Save final gallery state
        gallery.save(suffix="_final")
        
        # Convert annotated videos to H.264 using ffmpeg for web playback compatibility
        PIPELINE_STATUS["status"] = "converting"
        PIPELINE_STATUS["progress"] = 95.0
        PIPELINE_STATUS["current_step"] = "Finalizing: Encoding videos to web-compatible H.264 format..."
        
        cctv1_out = OUTPUT_DIR / "cctv-1_annotated.mp4"
        cctv2_out = OUTPUT_DIR / "cctv-2_annotated.mp4"
        cctv1_web = OUTPUT_DIR / "cctv-1_web.mp4"
        cctv2_web = OUTPUT_DIR / "cctv-2_web.mp4"
        
        # Run ffmpeg conversion
        if cctv1_out.exists():
            subprocess.run(["ffmpeg", "-y", "-i", str(cctv1_out), "-vcodec", "libx264", "-acodec", "aac", str(cctv1_web)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if cctv2_out.exists():
            subprocess.run(["ffmpeg", "-y", "-i", str(cctv2_out), "-vcodec", "libx264", "-acodec", "aac", str(cctv2_web)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        PIPELINE_STATUS["status"] = "completed"
        PIPELINE_STATUS["progress"] = 100.0
        PIPELINE_STATUS["current_step"] = "Processing completed successfully!"
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.exception("Error running ReID pipeline")
        PIPELINE_STATUS["status"] = "failed"
        PIPELINE_STATUS["error"] = str(e)
        PIPELINE_STATUS["current_step"] = f"Failed: {str(e)}"

@app.post("/api/upload")
async def upload_videos(
    cctv1: UploadFile = File(...),
    cctv2: UploadFile = File(...)
):
    global PIPELINE_STATUS
    if PIPELINE_STATUS["status"] == "running" or PIPELINE_STATUS["status"] == "converting":
        return JSONResponse(status_code=400, content={"error": "Pipeline is already running."})
        
    # Save uploaded files
    cctv1_path = UPLOAD_DIR / "cctv1.mp4"
    cctv2_path = UPLOAD_DIR / "cctv2.mp4"
    
    try:
        with cctv1_path.open("wb") as buffer:
            shutil.copyfileobj(cctv1.file, buffer)
        with cctv2_path.open("wb") as buffer:
            shutil.copyfileobj(cctv2.file, buffer)
            
        return {"message": "Files uploaded successfully."}
    except Exception as e:
        logger.exception("Error saving uploaded files")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start")
def start_pipeline(background_tasks: BackgroundTasks):
    global PIPELINE_STATUS
    if PIPELINE_STATUS["status"] == "running" or PIPELINE_STATUS["status"] == "converting":
        return JSONResponse(status_code=400, content={"error": "Pipeline is already running."})
        
    cctv1_path = UPLOAD_DIR / "cctv1.mp4"
    cctv2_path = UPLOAD_DIR / "cctv2.mp4"
    
    if not cctv1_path.exists() or not cctv2_path.exists():
        return JSONResponse(status_code=400, content={"error": "Please upload both CCTV videos first."})
        
    background_tasks.add_task(run_reid_pipeline_bg, str(cctv1_path), str(cctv2_path))
    return {"message": "Pipeline started in background."}

@app.get("/api/status")
def get_status():
    global PIPELINE_STATUS
    return PIPELINE_STATUS

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    dashboard_path = PROJECT_DIR / "dashboard.html"
    if dashboard_path.exists():
        with open(dashboard_path, "r") as f:
            return f.read()
    else:
        raise HTTPException(status_code=404, detail="Dashboard UI not found")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
