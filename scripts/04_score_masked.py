#!/usr/bin/env python3

import cv2
import numpy as np
import os
import json
import math
import re
import imagehash
from pathlib import Path
from shutil import copy2, rmtree
from tqdm import tqdm
from collections import defaultdict
from ultralytics import YOLO
import torch
from PIL import Image
import argparse

# --- Configuration ---
WEIGHTS = {
    "pose_score": 0.60,
    "sharpness_score": 0.30,
    "brightness_score": 0.10
}
IDEAL_BRIGHTNESS = 128.0

# De-duplication
TOP_N_IMAGES = 100
GLOBAL_HASH_THRESHOLD = 3
INTRA_SOURCE_HASH_THRESHOLD = 10

# YOLO/GPU Config
MODEL_NAME = 'yolov8l-pose.pt'
DEVICE = 0 

# --- QA FILTERS ---
# "Ottoman" Test: Max allowed separate large objects
MAX_CONTOUR_COUNT = 3

# --- NEW SHARPNESS FILTER (from user review) ---
# "Blurry" Test: Discard if below this raw sharpness score
MIN_SHARPNESS_THRESHOLD = 50.0
# --- END NEW FILTER ---

KEYPOINT_NAMES = {
    0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
    5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
}
# --- End Configuration ---

FACE_KEYPOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
UPPER_BODY_KEYPOINTS = FACE_KEYPOINTS + ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
FULL_BODY_KEYPOINTS = list(KEYPOINT_NAMES.values())

def get_source_video(filename):
    try:
        return filename.split("_frame_")[0]
    except IndexError:
        return "unknown_source"

def get_frame_index(filename):
    match = re.search(r'_frame_(\d+)\.(jpg|jpeg|png|webp)$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

def get_brightness_score(image_gray, mask):
    masked_pixels = image_gray[mask > 0]
    if masked_pixels.size == 0:
        return 0.0
    mean = np.mean(masked_pixels)
    score = 1.0 - (abs(mean - IDEAL_BRIGHTNESS) / IDEAL_BRIGHTNESS)
    return max(0, score)

def get_sharpness_score(image_gray, mask):
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    masked_laplacian = laplacian[mask > 0]
    if masked_laplacian.size == 0:
        return 0.0
    return masked_laplacian.var()

def get_pose_score(yolo_keypoints, category_name):
    if yolo_keypoints.shape[0] == 0: return 0.0
    conf_tensor = yolo_keypoints.conf[0]
    conf_map = {
        KEYPOINT_NAMES[i]: conf_tensor[i].item() 
        for i in range(len(KEYPOINT_NAMES))
    }
    get_conf = lambda name: conf_map.get(name, 0.0)

    if category_name == "face_and_hair":
        keypoints_to_check = FACE_KEYPOINTS
    elif category_name == "upper_body":
        keypoints_to_check = UPPER_BODY_KEYPOINTS
    elif category_name == "full_body":
        keypoints_to_check = FULL_BODY_KEYPOINTS
    else:
        return 0.0

    if not keypoints_to_check: return 0.0
    total_confidence = sum(get_conf(name) for name in keypoints_to_check)
    return total_confidence / len(keypoints_to_check)

def get_mask_score(mask):
    """
    Analyzes a mask for object count.
    Returns 1.0 (good) or 0.0 (bad).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # --- Test 1: Contour Count (The "Ottoman" Test) ---
    min_area = (mask.shape[0] * mask.shape[1]) * 0.005 
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    contour_count = len(large_contours)
    
    if contour_count > MAX_CONTOUR_COUNT:
        return 0.0, f"Fail: Too many objects ({contour_count})"
        
    if contour_count == 0:
        return 0.0, "Fail: Empty mask"

    # --- Test 2: Solidity (SKIPPED) ---
    # User review found this test was flagging good dynamic poses.
        
    return 1.0, f"Pass (C:{contour_count})"

def main():
    parser = argparse.ArgumentParser(description="Score and rank *masked* frames.")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        required=True, 
        help="The base project directory (e.g., /projects/tara_tainton_TEST)"
    )
    args = parser.parse_args()

    BASE_DIR = Path(args.base_dir)
    INPUT_DIR = BASE_DIR / "03_masked_subjects" 
    OUTPUT_DIR = BASE_DIR / "04_top100_staging" 
    OUTPUT_JSON_FILE = BASE_DIR / "scoring_results_yolo_masked.json"
    OUTPUT_REPORT_FILE = BASE_DIR / "ranked_report_yolo_masked.html"
    MODEL_PATH = BASE_DIR / MODEL_NAME 

    CATEGORIES_TO_SCORE = {
        "face_and_hair": INPUT_DIR / "face_and_hair",
        "upper_body": INPUT_DIR / "upper_body",
        "full_body": INPUT_DIR / "full_body"
    }

    print("--- Starting Phase 4: MASK-AWARE Smart Scorer (with QA Filters) ---")
    if not torch.cuda.is_available():
        print("\n*** FATAL ERROR: PyTorch cannot detect your GPU. ***"); return
    print(f"GPU Detected: {torch.cuda.get_device_name(DEVICE)}. Proceeding.")
    print(f"Mask QA Filters: Max Objects <= {MAX_CONTOUR_COUNT}")
    print(f"Sharpness Filter: Raw Sharpness >= {MIN_SHARPNESS_THRESHOLD}") # --- NEW PRINT ---


    if not MODEL_PATH.exists():
        print(f"\n*** FATAL ERROR: Model file not found at {MODEL_PATH} ***")
        return

    print(f"Loading model '{MODEL_PATH.name}'...")
    model = YOLO(str(MODEL_PATH))
    model.to(DEVICE)
    print("Model loaded successfully.")

    print("\n--- Pass 0: Building Video Manifest ---")
    video_manifest = defaultdict(int)
    MASTER_FRAME_DIR = BASE_DIR.parent / "01_source_frames"
    if not MASTER_FRAME_DIR.is_dir():
        MASTER_FRAME_DIR = BASE_DIR / "01_source_frames" # Fallback
        
    all_image_files = list(MASTER_FRAME_DIR.rglob("*.jpg")) + list(MASTER_FRAME_DIR.rglob("*.jpeg"))
    
    if not all_image_files:
        print(f"*** FATAL ERROR: No images found in {MASTER_FRAME_DIR}. ***")
        return

    for img_path in tqdm(all_image_files, desc="Scanning all source files"):
        source_video = get_source_video(img_path.name)
        frame_index = get_frame_index(img_path.name)
        if frame_index > video_manifest[source_video]:
            video_manifest[source_video] = frame_index
    print(f"Manifest complete. Found max frame indices for {len(video_manifest)} videos.")

    all_final_results = {}

    for cat_name, cat_path in CATEGORIES_TO_SCORE.items():
        print(f"\n--- Processing Category: {cat_name} ---")
        if not cat_path.is_dir():
            print(f"Warning: Folder not found {cat_path}. Skipping.")
            continue

        image_files = list(cat_path.glob("*.png"))
        
        if not image_files:
            print(f"No PNG images found in {cat_path}. Did 03_mask_subjects.py run?")
            continue

        all_category_scores = []
        
        for img_path in tqdm(image_files, desc=f"Scoring {cat_name} (Pass 1)"):
            try:
                image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if image is None: 
                    tqdm.write(f"Warning: Could not read {img_path.name}, skipping.")
                    continue
                
                if image.shape[2] != 4:
                    tqdm.write(f"Warning: {img_path.name} is not 4-channel. Skipping.")
                    continue
                    
                b, g, r, a = cv2.split(image)
                mask = a # The alpha channel is our mask
                
                image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

                # --- Run Scorers & Filters ---
                
                # --- FILTER 1: MASK QA ---
                raw_mask_score, mask_status = get_mask_score(mask)
                if raw_mask_score == 0.0:
                    all_category_scores.append({
                        "file": str(img_path.name), "path": str(img_path), "phash": "N/A",
                        "source_video": get_source_video(img_path.name),
                        "frame_index": get_frame_index(img_path.name),
                        "scores": {
                            "raw_brightness": 0, "raw_sharpness": 0, "raw_pose": 0,
                            "raw_mask_score": 0.0, "mask_status": mask_status
                        },
                        "final_score": 0.0 # Discard
                    })
                    continue # Skip to the next image
                
                # --- FILTER 2: SHARPNESS QA ---
                raw_sharpness = get_sharpness_score(image_gray, mask)
                if raw_sharpness < MIN_SHARPNESS_THRESHOLD:
                    all_category_scores.append({
                        "file": str(img_path.name), "path": str(img_path), "phash": "N/A",
                        "source_video": get_source_video(img_path.name),
                        "frame_index": get_frame_index(img_path.name),
                        "scores": {
                            "raw_brightness": 0, "raw_sharpness": raw_sharpness, "raw_pose": 0,
                            "raw_mask_score": 1.0, "mask_status": "Fail: Too blurry"
                        },
                        "final_score": 0.0 # Discard
                    })
                    continue # Skip to the next image

                # --- IMAGE PASSED ALL FILTERS ---
                
                # 1. Pose
                results = model(str(img_path), device=DEVICE, verbose=False)
                raw_pose = get_pose_score(results[0].keypoints, cat_name)
                
                # 2. Brightness
                raw_brightness = get_brightness_score(image_gray, mask)
                
                # 3. pHash
                image_pil = Image.open(img_path)
                phash = str(imagehash.phash(image_pil))
                
                all_category_scores.append({
                    "file": str(img_path.name), "path": str(img_path),
                    "phash": phash, "source_video": get_source_video(img_path.name),
                    "frame_index": get_frame_index(img_path.name),
                    "scores": {
                        "raw_brightness": raw_brightness,
                        "raw_sharpness": raw_sharpness, # We already calculated this
                        "raw_pose": raw_pose,
                        "raw_mask_score": raw_mask_score,
                        "mask_status": mask_status
                    }
                })
            except Exception as e:
                tqdm.write(f"Error on {img_path.name}: {e}")

        if not all_category_scores:
            print("No valid images scored in this category. Skipping.")
            continue

        valid_scores = [item for item in all_category_scores if item.get('final_score', 1.0) != 0.0]
        if not valid_scores:
            print("No images passed the QA filters in this category. Skipping.")
            continue

        print(f"Normalizing {len(valid_scores)} images (Pass 2)...")
        max_sharpness = max(item['scores']['raw_sharpness'] for item in valid_scores)
        if max_sharpness == 0: max_sharpness = 1.0

        for item in all_category_scores:
            if item.get('final_score', 1.0) == 0.0:
                continue

            S_sharpness = item['scores']['raw_sharpness'] / max_sharpness
            item['scores']['S_sharpness'] = S_sharpness
            S_brightness = item['scores']['raw_brightness']
            S_pose = item['scores']['raw_pose']
            
            item['final_score'] = (
                S_pose * WEIGHTS['pose_score'] +
                S_sharpness * WEIGHTS['sharpness_score'] +
                S_brightness * WEIGHTS['brightness_score']
            )

        all_category_scores.sort(key=lambda x: x["final_score"], reverse=True)

        print(f"Building unique top {TOP_N_IMAGES} list (3-Stage Filter)...")
        top_n_items = []
        seen_global_hashes = set()
        seen_source_hashes = defaultdict(list)
        seen_source_frame_indices = defaultdict(list)

        is_face_category = (cat_name == "face_and_hair")

        for item in tqdm(all_category_scores, desc=f"Filtering unique {cat_name}"):
            if len(top_n_items) >= TOP_N_IMAGES: break
            
            if item['final_score'] == 0.0:
                continue
                
            current_hash = imagehash.hex_to_hash(item["phash"])
            current_source_video = item["source_video"]

            is_global_dupe = False
            for seen_hash in seen_global_hashes:
                if (current_hash - seen_hash) < GLOBAL_HASH_THRESHOLD:
                    is_global_dupe = True; break
            if is_global_dupe: continue

            is_source_dupe = False
            for seen_hash in seen_source_hashes[current_source_video]:
                if (current_hash - seen_hash) < INTRA_SOURCE_HASH_THRESHOLD:
                    is_source_dupe = True; break
            if is_source_dupe: continue

            if not is_face_category:
                selected_indices = seen_source_frame_indices[current_source_video]
                selected_count = len(selected_indices)
                total_frames = video_manifest.get(current_source_video, 0) 

                if total_frames == 0: continue

                if selected_count == 5:
                    avg_index = sum(selected_indices) / 5
                    midpoint = total_frames / 2
                    current_index = item["frame_index"]
                    test_passed = False
                    if avg_index < midpoint:
                        if current_index > (avg_index + total_frames) / 2: test_passed = True
                    else:
                        if current_index < (avg_index / 2): test_passed = True
                    if not test_passed: continue

                elif selected_count >= 6:
                    continue

            top_n_items.append(item)
            seen_global_hashes.add(current_hash)
            seen_source_hashes[current_source_video].append(current_hash)
            if not is_face_category:
                seen_source_frame_indices[current_source_video].append(item["frame_index"])
        
        all_final_results[cat_name] = all_category_scores 
        
        dest_folder = OUTPUT_DIR / cat_name

        print(f"Copying {len(top_n_items)} unique top images to {dest_folder}...")
        if dest_folder.exists(): rmtree(dest_folder)
        dest_folder.mkdir(parents=True)

        for i, item in enumerate(top_n_items):
            rank = i + 1
            original_path = Path(item["path"])
            new_filename = f"{rank:03d}_{original_path.name}"
            dest_path = dest_folder / new_filename
            copy2(original_path, dest_path)
            
            txt_path = original_path.with_suffix(".txt")
            if txt_path.exists():
                copy2(txt_path, dest_path.with_suffix(".txt"))

    print(f"\nSaving detailed scoring report to {OUTPUT_JSON_FILE}...")
    try:
        with open(OUTPUT_JSON_FILE, 'w') as f:
            json.dump(all_final_results, f, indent=2)
    except Exception as e:
        print(f"Error saving JSON report: {e}")

    print(f"Generating simple HTML report at {OUTPUT_REPORT_FILE}...")
    generate_html_report(all_final_results, OUTPUT_REPORT_FILE, BASE_DIR) 

    print("\n--- Phase 4 Scoring Complete ---")
    print(f"Top unique images saved to: {OUTPUT_DIR}")
    print(f"To view the report, exit the container and open this file on your host:")
    print(f"file://{OUTPUT_REPORT_FILE.absolute()}")


def generate_html_report(data, output_path, base_project_dir):
    html_content = """
    <html><head><title>YOLO Masked-Ranked Image Report (QA-Filtered)</title>
    <style>
        body { font-family: sans-serif; background: #222; color: #eee; margin: 0; padding: 0; }
        h1 { text-align: center; border-bottom: 2px solid #555; padding-bottom: 10px; }
        h2 { border-bottom: 1px solid #444; padding: 10px; background: #333; }
        .container { width: 98%; margin: auto; }
        .category { margin-bottom: 50px; }
        .item-grid { display: flex; flex-wrap: wrap; justify-content: center; }
        .item {
            display: inline-block; width: 250px; margin: 10px;
            background: #333; border-radius: 8px; overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3); vertical-align: top;
            border: 1px solid #444;
        }
        .item.fail {
            border: 2px solid #f77;
            opacity: 0.6;
        }
        .item img { 
            width: 100%; height: 250px; object-fit: cover; 
            background-image: 
                linear-gradient(45deg, #444 25%, transparent 25%), 
                linear-gradient(-45deg, #444 25%, transparent 25%), 
                linear-gradient(45deg, transparent 75%, #444 75%), 
                linear-gradient(-45deg, transparent 75%, #444 75%);
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
        }
        .item p { margin: 10px; font-size: 0.8em; word-wrap: break-word; }
        .item b { color: #8af; }
        .item .fail-reason { color: #f88; font-weight: bold; }
    </style>
    </head><body><div class="container"><h1>YOLO Masked-Ranked Image Report (QA-Filtered)</h1>
    """

    for category_name, items in data.items():
        html_content += f"<div class='category'><h2>Category: {category_name} (Top {min(len(items), 200)} Ranked)</h2>"
        html_content += "<div class='item-grid'>"

        for i, item in enumerate(items[:200]): 
            rank = i + 1
            score = item.get('final_score', 0)
            file_name = item.get('file', 'N/A')

            try:
                img_path = Path(item.get('path', ''))
                relative_img_path = img_path.relative_to(base_project_dir)
            except ValueError:
                relative_img_path = "path_error"

            s = item['scores']
            
            item_class = ""
            mask_status_html = ""
            if s.get('raw_mask_score', 1.0) == 0.0:
                item_class = "fail"
                mask_status_html = f"<b class='fail-reason'>Mask QA: {s.get('mask_status', 'FAIL')}</b><br>"
            # --- NEW: Check for blurriness fail ---
            elif s.get('mask_status', '') == 'Fail: Too blurry':
                item_class = "fail"
                mask_status_html = f"<b class='fail-reason'>Sharpness QA: {s['raw_sharpness']:.0f} (Too Blurry)</b><br>"
            else:
                mask_status_html = f"<b>Mask QA:</b> {s.get('mask_status', 'N/A')}<br>"
            
            details = (
                f"<b>Rank:</b> {rank}<br>"
                f"<b>Final Score:</b> {score:.4f}<br>"
                f"<hr><b>File:</b> {file_name}<br>"
                f"<b>Source:</b> {item['source_video']}<br>"
                f"<b>Frame:</b> {item['frame_index']}<br>"
                f"<hr><b>S_Pose:</b> {s['raw_pose']:.3f} (w: {WEIGHTS['pose_score']})<br>"
                f"<b>S_Sharp (Masked):</b> {s.get('S_sharpness', 0):.3f} (raw: {s['raw_sharpness']:.0f})<br>"
                f"<b>S_Bright (Masked):</b> {s['raw_brightness']:.3f}<br>"
                f"{mask_status_html}"
            )

            html_content += (
                f"<div class='item {item_class}'>" 
                f"<img src='{relative_img_path}' alt='{file_name}' loading='lazy'>"
                f"<p>{details}</p>"
                f"</div>"
            )

        html_content += "</div></div>"

    html_content += "</div></body></html>"

    try:
        output_path.write_text(html_content)
    except Exception as e:
        print(f"Error writing HTML report: {e}")

if __name__ == "__main__":
    main()
