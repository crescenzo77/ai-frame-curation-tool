# AI-Powered Dataset Curation Pipeline

This project is a complete, containerized pipeline for processing a large, raw pool of video frames into a small, high-quality, and varied dataset for AI model training (e.g., SDXL LoRA).

It automatically sifts, sorts, scores, and de-duplicates images to find the best available frames, solving common dataset problems like poor focus, bad framing, and low variety.

## Key Features

* **Native Frame Extraction:** Extracts frames at their original, native resolution, preserving aspect ratio for multi-aspect training.
* **2-Stage Smart Sorting:** A highly-configurable `02_sort_dataset.py` script first runs a "permissive" filter to categorize images (face, upper body, full body) and filter out obvious "no-person" shots.
* **Advanced Focus Scoring:** `03_score_and_rank_v3.py` implements a "v4" scoring model that uses 3 separate ML models (Holistic, Selfie Segmentation, Pose) to find images where the subject's face is the sharpest object in the frame, penalizing images with blurry faces but sharp backgrounds.
* **3-Stage De-duplication Gauntlet:** Ensures a high-variety dataset by combining three filters:
    * **Global pHash:** A strict filter to remove near-identical twins.
    * **Intra-Source pHash:** A laxer filter to ensure visual variety from the *same* video.
    * **Temporal Quota:** A configurable hard cap (e.g., 6 frames) per video to force selections from different timeline segments.
* **Automated Captioning:** Includes scripts to auto-caption the final dataset using the Gemini or Vertex AI APIs.
* **Containerized & Portable:** The entire environment (OpenCV, MediaPipe, PyTorch) is containerized with Docker, making it 100% reproducible.

---

## The Pipeline: How It Works

The scripts are numbered in the order you should run them.

1.  **`01_extract_frames.py`:**
    * **Input:** Raw video files in `/projects/source_videos`.
    * **Output:** Native-resolution frames in `/projects/source_images`, organized into subfolders.

2.  **`02_sort_dataset.py`:**
    * **Input:** All images in `/projects/source_images`.
    * **Output:** Images are sorted by pose/shot type (face, upper_body, full_body) into `/projects/sorted_output`. This script's thresholds (like `MIN_POSE_AREA_PERCENT`) can be tuned to be more or less permissive.

3.  **`03_score_and_rank_v3.py`:**
    * This is the core logic.
    * **Input:** The sorted images from Step 2.
    * **Process:** Runs the v4 "Focus-Weighted" scoring and the "3-Stage" de-duplication filter on all candidate images. Saves a complete `scoring_results_v3.json` file.
    * **Output:** The **Top N** (e.g., Top 60) best, most unique images for each category, copied to new folders (e.g., `/projects/sorted_output/a_face_n_hair_top60_v4_temporal`).

4.  **`04_caption_dataset_*.py`:**
    * **Input:** The final "Top N" folders.
    * **Process:** Generates a descriptive caption for each image and saves it as a `.txt` file.
    * **Output:** A complete, captioned, high-quality dataset ready for AI training.

*(Your `docker-compose.yml`, `Dockerfile`, etc. in the repo should already be clean and correct.)*
