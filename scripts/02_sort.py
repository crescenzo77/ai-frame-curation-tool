#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import torch
import argparse # <-- 1. IMPORTED

# --- Configuration ---
# 2. Hard-coded paths are REMOVED
MODEL_NAME = 'yolov8l-pose.pt'
CONF_THRESHOLD = 0.5
DEVICE = 0

KEYPOINT_NAMES = {
    0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
    5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
}
# --- End Configuration ---

def main():
    # 3. ADDED ARGUMENT PARSER
    parser = argparse.ArgumentParser(description="Sort frames using YOLO-Pose.")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        required=True, 
        help="The base project directory (e.g., /projects/tara_tainton)"
    )
    args = parser.parse_args()

    # 4. DEFINE PATHS RELATIVE TO base_dir
    BASE_DIR = Path(args.base_dir)
    
    # --- 5. FIX THE PATHS ---
    # The output of step 1 is our input
    SOURCE_DIR = BASE_DIR / "01_source_frames" 
    # The output for this script
    OUTPUT_DIR = BASE_DIR / "02_sorted_candidates"
    # The model file, which lives in the base directory
    MODEL_PATH = BASE_DIR / MODEL_NAME 
    # --- End Fix ---

    # 6. Categories are now relative to the new OUTPUT_DIR
    CATEGORIES = {
        "face_and_hair": OUTPUT_DIR / "face_and_hair",
        "upper_body": OUTPUT_DIR / "upper_body",
        "full_body": OUTPUT_DIR / "full_body",
        "review_no_face": OUTPUT_DIR / "review_no_face",
        "no_person_detected": OUTPUT_DIR / "no_person_detected",
        "uncategorized": OUTPUT_DIR / "uncategorized"
    }

    print("--- Starting Phase 2: YOLO Smart Sorter ---")

    if not torch.cuda.is_available():
        print("\n*** FATAL ERROR: PyTorch cannot detect your GPU. ***")
        print("Make sure you are running this inside the Docker container.")
        return
    print(f"GPU Detected: {torch.cuda.get_device_name(DEVICE)}. Proceeding.")

    print(f"Creating output directories in: {OUTPUT_DIR}")
    for path in CATEGORIES.values():
        path.mkdir(parents=True, exist_ok=True)

    # 7. CHECK for model file
    if not MODEL_PATH.exists():
        print(f"\n*** FATAL ERROR: Model file not found at {MODEL_PATH} ***")
        print(f"Please ensure '{MODEL_NAME}' is in your base project directory: {BASE_DIR}")
        return
        
    print(f"Loading model '{MODEL_PATH.name}'...")
    model = YOLO(str(MODEL_PATH)) # <-- Load from specific path
    model.to(DEVICE)
    print("Model loaded successfully.")

    image_files = list(SOURCE_DIR.rglob("*.jpg")) + \
                  list(SOURCE_DIR.rglob("*.jpeg")) + \
                  list(SOURCE_DIR.rglob("*.png")) + \
                  list(SOURCE_DIR.rglob("*.webp"))

    if not image_files:
        print(f"\n*** FATAL ERROR: No images found in {SOURCE_DIR} ***")
        print("Did Phase 1 (extraction & validation) complete successfully?")
        return

    print(f"Found {len(image_files)} images. Starting processing...")
    
    # --- This is the sorting logic ---
    for img_path in tqdm(image_files, desc="Sorting image batch"):
        try:
            results = model(str(img_path), device=DEVICE, verbose=False)

            if results[0].keypoints.shape[0] == 0:
                shutil.copy(img_path, CATEGORIES["no_person_detected"])
                continue

            conf_tensor = results[0].keypoints.conf[0]
            conf_map = {
                KEYPOINT_NAMES[i]: conf_tensor[i].item() 
                for i in range(len(KEYPOINT_NAMES))
            }
            get_conf = lambda name: conf_map.get(name, 0.0)

            has_face = get_conf('nose') > CONF_THRESHOLD and get_conf('left_eye') > CONF_THRESHOLD
            has_shoulders = get_conf('left_shoulder') > CONF_THRESHOLD or get_conf('right_shoulder') > CONF_THRESHOLD
            has_hips = get_conf('left_hip') > CONF_THRESHOLD or get_conf('right_hip') > CONF_THRESHOLD
            has_ankles = get_conf('left_ankle') > CONF_THRESHOLD or get_conf('right_ankle') > CONF_THRESHOLD

            target_folder = None
            if has_face and has_shoulders and not has_hips:
                target_folder = CATEGORIES["face_and_hair"]
            elif has_face and has_shoulders and has_hips and not has_ankles:
                target_folder = CATEGORIES["upper_body"]
            elif has_face and has_shoulders and has_hips and has_ankles:
                target_folder = CATEGORIES["full_body"]
            elif (not has_face) and has_shoulders:
                target_folder = CATEGORIES["review_no_face"]
            else:
                target_folder = CATEGORIES["uncategorized"]

            shutil.copy(img_path, target_folder)

        except Exception as e:
            tqdm.write(f"Error on {img_path.name}: {e}")

    print("\n--- Phase 2 Sorting Complete ---")
    print(f"Sorted images are in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
