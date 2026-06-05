# Multi‑Camera Person Re‑Identification System

## 1. Problem Statement
Build a **multi‑camera person tracking system** that assigns a **unique, persistent global ID** to each person detected across multiple CCTV feeds **without any prior identity database, names, or registered profiles**.  
The system must perform **anonymous re‑identification (ReID)** using only visual features (clothing, body shape), movement patterns, and temporal consistency.  
Example:  
- CCTV‑1 detects Person A → ID‑01  
- Later, CCTV‑2 sees the same Person A → still labelled ID‑01  

**Core challenge**: zero‑shot, open‑set identity association across non‑overlapping cameras with varying viewpoints, lighting, and occlusions.

---

## 2. System Overview (5‑Layer Pipeline)
```
[Detection] → [Single‑Cam Tracking] → [Feature Extraction] → [Cross‑Cam Matching] → [Global ID Management]
```
- **Person Detection** – locate every person in each frame.  
- **Single‑Camera Tracking (SCT)** – maintain local IDs while a person is visible in one camera.  
- **Feature Extraction** – convert each person’s appearance into a discriminative embedding.  
- **Cross‑Camera Matching** – compare embeddings against a global gallery using spatio‑temporal constraints.  
- **Global ID Assignment** – create new IDs or reuse existing ones via online clustering.

---

## 3. Tech Stack & Tools (With Justification)

| Component | Library / Model | Purpose |
|-----------|----------------|---------|
| **Person Detection** | YOLOv8n (ultralytics) | Fast, accurate, class=0 (person), easy integration |
| **Single‑Cam Tracking** | ByteTrack (integrated with Ultralytics) | Strong occlusion handling, doesn’t rely on appearance (smooth hand‑off between detection and ReID) |
| **Feature Extraction** | **OSNet‑x1.0** (torchreid) or **TransReID** | Pre‑trained on Market‑1501 / MSMT17; outputs 512‑d L2‑normalised embedding robust to clothing and viewpoint changes |
| **GPU Acceleration** | PyTorch + CUDA / TensorRT | Inference speed for real‑time use |
| **Vector Search** | FAISS (faiss‑gpu or faiss‑cpu) | Fast cosine similarity search among thousands of gallery embeddings |
| **Batch Matching** | SciPy (linear_sum_assignment) | Hungarian algorithm for optimal one‑to‑one assignment when multiple people appear simultaneously |
| **Image Processing** | OpenCV (cv2) | Cropping, resizing, optional colour histogram extraction |
| **Camera Streaming** | OpenCV VideoCapture / RTSP | Handles real‑time feeds |
| **Spatio‑Temporal Model** | Custom (SciPy stats) | Gaussian fit on travel‑time distributions (μ, σ) per camera pair |
| **Gallery Persistence (Optional)** | ChromaDB / SQLite / Redis | If cross‑session memory is needed; otherwise in‑memory dictionary with FAISS index |
| **Orchestration** | Python threading / multiprocessing | One thread per camera for parallel processing |

---

## 4. Feature Extraction Details
The primary feature is a **deep appearance embedding**. Optional supplementary features can be added for extra robustness.

### 4.1 Primary: Deep Appearance Embedding
- **Model**: OSNet‑x1.0 (pretrained).  
- **Input**: Person crop resized to 256×128 (RGB).  
- **Output**: 512‑dimensional float32 vector, L2‑normalised.  
- **What it captures internally** (learned, not hand‑crafted):  
  - Clothing colour, texture, style  
  - Body silhouette / proportions  
  - Presence / position of accessories (bags, hats)  
  - Hairstyle, if visible  
- **Why this works**: OSNet was trained with metric learning (triplet loss) to pull same‑person crops together and push different ones apart across varying cameras.

**Representation of a tracklet**:  
For a person tracked over N frames in one camera, we take the **mean embedding** of all quality‑filtered crops (or keep a single best‑quality crop) and treat that as the tracklet’s signature. Gallery entries are updated with exponential moving average (EMA) when new observations arrive.

### 4.2 Supplementary Features (Optional – for hard cases)
| Feature | Dim | When useful |
|---------|-----|-------------|
| HSV colour histogram (upper/lower body split) | 136‑d | Uniformed staff, fast pre‑filter |
| Body‑part embeddings (4 horizontal zones) | 2048‑d | Heavy occlusion, extreme viewpoint change |
| Pose / skeleton ratios (YOLOv8‑pose) | 55‑d | Long‑term, clothing changes (biometric) |
| Gait / optical flow | 8‑128‑d | Appearance‑independent identification |
| Spatio‑temporal travel time | N/A (probability) | Multi‑camera layout, reduces false positives |

In the baseline system, only the 512‑d embedding is used; others are add‑ons for specific failure modes.

---

## 5. Gallery Management & Storage
The **Global Gallery** is the persistent memory of all observed persons.

### 5.1 In‑Memory Structure (Real‑Time)
```python
gallery = {
    "ID-001": {
        "embedding": np.array(512),      # current prototype (EMA)
        "first_seen": timestamp,
        "last_seen": timestamp,
        "cameras": {"CCTV-1", "CCTV-2"}
    },
    "ID-002": {...}
}
```
- **Prototype update**: `new_prototype = α * new_obs + (1-α) * old_prototype` (α=0.1), then re‑normalise.  
- **Matching**: cosine similarity between query embedding and all gallery prototypes.  
- **Search speed**: For >1000 entries, use **FAISS** `IndexFlatIP` (inner product ≈ cosine after normalisation).

### 5.2 Persistent Storage (Cross‑Session)
- **Option A**: Save gallery as pickle + FAISS index file (simple).  
- **Option B**: ChromaDB or SQLite for metadata + FAISS for vectors (scalable).  
- **Pruning**: Remove entries with `last_seen` older than a defined lifespan (e.g., 1 hour for same‑day tracking).

---

## 6. Spatio‑Temporal Model (Travel‑Time Constraints)
A learned probabilistic model filters impossible candidate matches.

### 6.1 Learning the Topology
- From unlabelled data: collect all transitions of **any** person between camera pairs.  
- For each camera pair (Cᵢ, Cⱼ), build a **Gaussian distribution** over the time gap Δt = t_appearⱼ - t_disappearᵢ.  
- If no data exists, start with manually defined min/max seconds.

### 6.2 Real‑Time Gating
When a new person appears in camera C at time t:
- For each existing global ID, check feasibility:  
  `p(Δt | last_cam, C)` > threshold (e.g., within μ±3σ).  
- If not feasible, exclude that ID from the candidate list **before** computing appearance similarity.

This alone eliminates absurd matches (e.g., a person exiting the front door cannot instantly reappear on the 5th floor).

---

## 7. Edge Cases & Mitigations

### 7.1 Time Travel (Teleportation) / Unrealistic Transitions
**Scenario**: Person leaves camera A at t=0s, appears in camera B at t=1s when minimum walk time is 15s.  
**Mitigation**: Spatio‑temporal gate (Section 6). If the time gap is outside the learned Gaussian’s high‑probability region, the match is rejected. This prevents false ID preservation.

### 7.2 Clothing Changes (Same Day / Different Day)
**Scenario**: Person puts on a jacket or leaves a bag behind.  
**Impact**: Deep appearance embedding shifts, similarity drops below threshold.  
**Mitigation**:  
- Use body‑part features (upper/lower body separately) – a jacket change mainly affects upper body, legs may still match.  
- EMA update: if the change occurs gradually across camera views, gallery prototype adapts.  
- For long‑term (cross‑day), **fuse with gait or face** (if policy allows). Without biometrics, true identity across clothing changes remains an open problem; system may assign a new ID.  
- Short‑term (within minutes): spatio‑temporal constraints often save the match because only one person can occupy that path at that exact time.

### 7.3 Crowded Scenes / Groups of People
**Scenario**: 5 people enter a room simultaneously.  
**Challenges**: Heavy occlusion, multiple similar appearances (uniforms), local tracker may fragment.  
**Mitigation**:
- **Detection**: YOLO + ByteTrack handle moderate occlusion; low‑confidence detections are retained.  
- **Batch matching with Hungarian algorithm**: When a group arrives, a cost matrix (1‑cosine) between all new tracks and all valid gallery entries is minimised, avoiding greedy assignment errors.  
- **Hard‑negative mining**: Use body‑part embeddings to distinguish uniforms.  
- **Spatio‑temporal correlation**: People moving as a group often exit the previous camera together; the travel times cluster closely, providing a soft group constraint.

### 7.4 Occlusions and Partial Views
**Crop quality**: If a person is heavily occluded (e.g., only head visible), the embedding may be noisy.  
**Mitigation**: Skip detections with very small area (<20px height). Use tracklet‑level aggregation (mean of multiple frames) to filter out noise.

### 7.5 Lighting / Viewpoint Drift
**Mitigation**:  
- Pre‑processing: histogram equalisation.  
- The ReID model is trained with heavy data augmentation (colour jitter, random erasing).  
- Gallery EMA updates continuously adapt the prototype to new camera views.

---

## 8. Validation & Evaluation Strategy
Since we have **no labelled data** from our own cameras, we validate in two stages:

### 8.1 Offline Benchmarking (Before Deployment)
- Use public datasets (Market‑1501, DukeMTMC) to measure:  
  - **Rank‑1 accuracy** – is the correct person top‑1 in the gallery?  
  - **mAP** – mean Average Precision across queries.  
- **Threshold calibration**: Plot precision‑recall curve to choose the best cosine threshold (e.g., 0.65–0.75) for our desired trade‑off (higher threshold = fewer false merges, more false splits).

### 8.2 On‑Site Validation (Without Ground Truth)
- **Manual spot checks**: Security personnel review short clips with overlaid IDs and confirm whether the same person keeps the same ID across cameras.  
- **Intrusion tests**: A volunteer wearing a known outfit walks a predefined route; we verify ID persistence and timing.  
- **Key metrics (computed from tracker output)**:  
  - **ID switches per person** (should be near zero for same‑camera, low for cross‑camera).  
  - **Fragment rate** (how often a single true person is split into multiple IDs).  
  - **Merge rate** (how often two different people get the same ID).  

### 8.3 Continuous Monitoring
- Deploy a dashboard tracking:  
  - Average gallery size  
  - Match confidence distribution  
  - Rate of new ID creation vs. re‑identification  
- If the percentage of very‑high‑confidence (>0.9) matches drops, thresholds may need adjustment.

---

## 9. Real‑Time Example Walkthrough (ML Perspective)
**Setup**: CCTV‑1 (entrance), CCTV‑2 (lobby). Travel time 5–15s.

1. **T=0s**: Person A enters CCTV‑1.  
   - YOLO detects → ByteTrack assigns local ID=1.  
   - ReID model extracts embedding `f_A`.  
   - Global gallery empty → **new cluster ID‑001** created (prototype = `f_A`).

2. **T=8s**: Person A leaves CCTV‑1.  
   - ID‑001’s `last_seen` updated to 8s, camera=CCTV‑1.

3. **T=10s**: Person B enters CCTV‑1.  
   - Embedding `f_B`.  
   - Cosine similarity with ID‑001 = 0.23 (< threshold 0.7) → **new cluster ID‑002**.

4. **T=13s**: Person A appears in CCTV‑2.  
   - Embedding `f_A2`.  
   - Spatio‑temporal gate: ID‑001 (last CCTV‑1 at 8s) can reach CCTV‑2 by 13s? µ=10s, σ=2s → within µ±3σ → feasible. ID‑002 also feasible.  
   - Cosine similarity: `f_A2` vs ID‑001 = 0.91, vs ID‑002 = 0.28.  
   - **Match → ID‑001 assigned**. Prototype updated: 0.9*old + 0.1*f_A2.

5. **T=18s**: Person B in CCTV‑2.  
   - Embedding `f_B2`.  
   - Sim(f_B2, ID‑001)=0.15, Sim(f_B2, ID‑002)=0.88 → ID‑002 assigned.

**Result**: Two people, each with one unique global ID across two cameras, zero false merges/splits.

---

## 10. Project Structure
```
multi_camera_reid/
├── reid_extractor.py       # OSNet wrapper, extract() method
├── gallery.py              # GlobalGallery class, EMA, batch_match, Hungarian
├── camera_pipeline.py      # Single camera: read, detect, track, extract, annotate
├── orchestrator.py         # Spawn threads, shared ReID extractor & gallery
├── st_model.py             # Spatio-temporal Gaussian model
├── utils.py                # Color histograms, part features, etc.
├── config.yaml             # Thresholds, camera URLs, topology
├── data/
│   └── test_videos/        # Sample videos for validation
├── logs/                   # Runtime logs
└── PROJECT.md              # This document
```

---

## 11. Key Parameters & Their Impact
| Parameter | Typical Value | Effect of Increase |
|-----------|---------------|-------------------|
| Cosine threshold | 0.65 – 0.75 | Fewer false merges, more fragments (new IDs) |
| EMA alpha (α) | 0.1 | Slower adaptation to appearance changes, more stable prototype |
| Min crop height | 20 px | Prevents noisy embeddings from tiny detections |
| Spatio‑temporal gate width | μ ± 3σ | Tighter gating → fewer candidates → fewer false positives, but might miss valid late arrivals |
| Gallery entry TTL (lifespan) | 3600 sec (1 hr) | Longer TTL enables re‑id after long absences but increases gallery size and risk of stale prototypes |

---

## 12. Limitations & Future Improvements
- **Cross‑day re‑id** not solved without biometrics (gait / face).  
- **Identical uniforms** remain challenging; additional cues (body proportions, gait) are required.  
- **Unsupervised domain adaptation** (e.g., fine‑tuning OSNet on target‑camera data via pseudo‑label clustering) could boost performance.  
- **Distributed deployment**: multiple edge devices with central FAISS server.  
- **Active learning**: operator corrections fed back to fine‑tune the ReID model or adjust thresholds.

---

## 13. Getting Started
```bash
pip install ultralytics torchreid faiss-gpu scipy opencv-python pyyaml
# For CPU-only: pip install faiss-cpu
python orchestrator.py --config config.yaml
```

All models download automatically on first run. No manual annotation required.

---

**Bottom line**: The system is an **unsupervised, online clustering machine** powered by a pre‑trained metric‑learning backbone, guided by learned spatio‑temporal probabilities. It turns raw CCTV streams into a persistent, anonymous identity space without any pre‑existing database.
