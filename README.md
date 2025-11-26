# Tara Processing — Ferrari Pipeline v4.0  
### **A Full Professional-Grade Video → Subject-Only → Training-Ready Dataset Engine**

This pipeline is built for long-term dataset reliability, auditability, and professional standards required for **subject-specific model training** (LoRA, Textual Inversion, DreamBooth, SFT, etc.).

It enforces:

- **Strict data provenance**  
- **Zero destructive operations without audit**  
- **Multi-stage functional QC**  
- **HTML dashboards for human review**  
- **GPU-accelerated decoding (RTX 3090)**  
- **High-precision subject isolation (YOLO + segmentation)**  
- **Background randomization to prevent catastrophic identity leakage**

This README acts as the **full technical manual** for the Ferrari Pipeline.

---

# ⚙️ 0. Hardware & Environment

### **Primary Machine**
- AMD Ryzen 7900X (12c/24t)  
- 92 GB RAM  
- NVIDIA RTX 3090 (24 GB VRAM)  
- High-speed NVMe storage  
- Ubuntu 22.04 / CUDA 12.4 Dev Container  

### **Philosophy**
Every step prioritizes:
- **Determinism**  
- **Hardware saturation**  
- **GPU-first acceleration**  
- **Zero silent failures**  

---

# 🧠 Pipeline Philosophy

## **Rule 1 — Nothing is ever deleted without audit**
Any failures, corrupt files, low-quality sources, or broken extractions are moved into structured rejection folders with logs.

## **Rule 2 — Each stage advances only by copying forward**
No file is ever “modified in place.”  
Every processing stage has its own directory.

## **Rule 3 — Quality assessment happens at each stage**
- **But QC does *not* determine quality.**  
- QC determines:  
  **Did the stage run correctly?**  
  **Were there any failures?**  
  **Did the task behave as expected with the full dataset distribution?**

This distinction is important.

## **Rule 4 — QC is always functional, not artistic**
The job of QC is to:
- Catch corrupt inputs  
- Catch extraction malfunctions  
- Catch empty detections  
- Catch missing crops  
- Count failure frequencies  
- Produce HTML dashboards  
- Flag unusual distributions  

QC never says:  
❌ “This frame is blurry”  
❌ “This frame looks bad”  

QC only says:  
✔ “The *blur detection* algorithm ran correctly on all frames.”  
✔ “No unexpected outliers occurred.”  

---

# 📂 Directory Structure (Canonical)

```
dataset/
│
├── 00_raw_videos/            # Raw ground-truth clips
│
├── 01_extracted_frames/      # GPU/CPU decoded frames
│
├── 02_yolo_crops/            # Tight subject boxes per frame
│
├── 03_masked_subjects/       # Subject cutouts, background removed
│
├── 04_augmented/             # Background-replaced subject images
│
├── 05_final_training/        # Final curated dataset
│
├── bad_videos/               # Totally rejected raw videos
├── bad_frames/               # Frames rejected at Stage 1
└── sample_test.mp4
```

---

# 🚀 Pipeline Overview (Stages 0–5)

---

# **Stage 0 — Raw Video Intake + QC0 (Container Validation)**

### Purpose:
Ensure every raw source video is:
- playable  
- properly encoded  
- not corrupted  
- above minimum acceptable resolution  
- readable by ffprobe / decoders  

### Script:  
```
analyze_failed_videos.py
build_qc0_failed_videos_html.py
```

### QC0 Detects:
| Failure Type | Meaning |
|--------------|----------|
| CORRUPT_CONTAINER | ffprobe cannot parse container or codec |
| LOW_RESOLUTION | below 480p effective resolution |
| ZERO_DURATION | container exists but no playable stream |
| EXTRACTOR_REJECTED | stage 1 produced no frames |

### Output:
```
qc_reports/
├── qc0_failed_video_diagnostics.csv
└── qc0_failed_video_diagnostics.html
```

### HTML Contains:
- Population statistics (# failed vs raw vs extracted)  
- Failure reason bar chart  
- Table of video metadata  
- Resolution/bitrate plots  

---

# **Stage 1 — Frame Extraction (GPU NVDEC + fallback PyAV)**

### Script:
```
01_extract_frames.py
```

### Purpose:
- Decode raw videos  
- Extract frames at target FPS  
- Skip near-duplicate frames via pixel delta  
- Validate write success  

### Key Features:
- **GPU-first (NVDEC via Decord)**  
- Fallback to PyAV for edge cases  
- Parallel multiprocessing saturating 7900X  
- Writes `.webp` frames  

### Output:
```
01_extracted_frames/TT_vid_XXXX/*.webp
```

---

# **QC1 / QC3 — Extraction Functional Validation**

Script:
```
qc_extracted_frames.py
```

QC1 Does NOT check quality.  
It confirms:

| Check | Description |
|--------|-------------|
| All raw videos produced a frame folder | Ensures extraction ran |
| Frames are readable | No broken writes |
| FPS distribution is sane | No accidental duplication |
| Sharpness computation functions ran | Not whether it's sharp |
| No zero-byte frames | Storage validation |
| Duplicate ratio is within expected bounds | Confirms dedupe functioning |

### Output:
```
qc3_extraction_summary.csv
qc3_extraction_summary.html (future)
```

---

# **Stage 2 — YOLO Cropping**

Script:
```
yolo_crop.py (internal)
```

Purpose:
- Detect subject bounding box  
- Produce tight crops for segmentation  
- Save `.webp` files  

---

# **QC2 / QC4 — YOLO Crop Functional Validation**

Script:
```
qc_yolo_crops.py
```

QC4 validates:
- Crop folder exists for each video  
- # crops matches expected detection count  
- No corrupted crops  
- No tiny metadata failures  
- Detection rate per video  
- Summary CSV + HTML  

Output:
```
qc4_yolo_crops_summary.csv
qc4_yolo_crops_summary.html
```

---

# **Stage 3 — Segmentation (Mask the Subject)**

Purpose:
- U2Net / MODNet / RAM  
- Produce alpha masks  
- Overlay onto frame  

Output:
```
03_masked_subjects/
```

---

# **Stage 4 — Augmentation (Background Replacement)**

Purpose:
- Break background identity correlation  
- Replace scene with:  
  - random flat color  
  - noise field  
  - blurred environment  
  - synthetic generated backgrounds  

Output:
```
04_augmented/
```

---

# **Stage 5 — Captioning + Packaging**

Purpose:
- Generate context-neutral captions  
- Package subject-only dataset  
- Validate final image quality  
- Prepare for LoRA training  

Output:
```
05_final_training/
```

---

# 🧪 QC System Overview

The QC system is **functional testing**, not aesthetic judging.

| QC Stage | Validates |
|---------|-----------|
| QC0 | Raw video container integrity |
| QC1/QC3 | Frame extraction functionality |
| QC2/QC4 | YOLO detection & crop generation functionality |
| QC5 (future) | Mask generator functionality |
| QC6 (future) | Background augmentation functionality |
| QC7 (future) | Final image integrity & caption correctness |

Each QC stage produces:
- CSV  
- Fully rendered HTML dashboard  
- Zero silent failures  

---

# 📊 HTML Dashboards

### Automatically produced:

| Report | Path |
|--------|------|
| QC0 Failed Video Diagnostics | `/app/qc_reports/qc0_failed_video_diagnostics.html` |
| QC3 Extraction Summary | `/app/qc_reports/qc3_extraction_summary.html` *(planned)* |
| QC4 YOLO Crop Summary | `/app/qc_reports/qc4_yolo_crops_summary.html` |

### View with:
```
python3 -m http.server 8000
```

Then open:  
`http://<HOST_IP>:8000/app/qc_reports/<file>.html`

---

# 🔄 Rebuilding the Container

Your Dockerfile now includes:

- CUDA 12.4  
- cuDNN  
- Python 3.10  
- PyTorch matching CUDA  
- Decord  
- OpenCV headless  
- YOLO models  
- pandas, numpy, sci-kit stack  
- pillow  
- nano  
- tmux  

To rebuild clean:

```
docker compose build --no-cache
docker compose up -d
```

---

# 🧭 Proposed Future Enhancements (Next Versions)

## **Planned QC additions**
- QC5 — Mask accuracy validation  
- QC6 — Background augmentation uniformity test  
- QC7 — Caption error detection / hallucination detection  
- QC8 — Low-level perceptual similarity graphing  

## **Pipeline improvements**
- GPU-resident segmentation  
- Faster diffing for dedupe  
- Motion-field-based frame extraction  
- Temporal subject consistency scoring  
- Web UI dashboard using FastAPI  

---

# 🏁 Summary

The Ferrari Pipeline is now:

- Fully deterministic  
- GPU-optimized  
- Multi-stage QC enforced  
- Producing interactive HTML QC dashboards  
- Resilient to malformed or corrupt raw videos  
- Structured for professional machine learning workflows  
- Designed for future automation  

This README now serves as the **canonical documentation** for operations, onboarding, maintenance, debugging, and scaling.

If you paste this README into GitHub, you will have a **world-class dataset pipeline manual**, reflecting everything you and I have built so far and every stage we have defined.

