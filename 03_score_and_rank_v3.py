#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
import os
import json
import math
from pathlib import Path
from shutil import copy2, rmtree
from tqdm import tqdm
import imagehash
from PIL import Image
from collections import defaultdict
import re

# --- Configuration ---
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_selfie_segmentation = mp.solutions.selfie_segmentation
PoseLandmark = mp_pose.PoseLandmark

WEIGHTS = {
    "pose": 0.20,
    "brightness": 0.10,
    "absolute_face_sharpness": 0.35,
    "focus_score": 0.35
}

TOP_N_IMAGES = 60
GLOBAL_HASH_THRESHOLD = 3
INTRA_SOURCE_HASH_THRESHOLD = 10

IDEAL_BRIGHTNESS = 128.0
VISIBILITY_THRESHOLD = 0.5
GAUSS_MU = 1.0
GAUSS_SIGMA = 0.25

BASE_PROJECT_DIR = Path("/projects")
SORTED_OUTPUT_DIR = BASE_PROJECT_DIR / "sorted_output"
OUTPUT_JSON_FILE = BASE_PROJECT_DIR / "scoring_results_v3.json"
OUTPUT_FOLDER_SUFFIX = f"_top{TOP_N_IMAGES}_v4_temporal"

CATEGORIES = {
    "a_face_n_hair": SORTED_OUTPUT_DIR / "a_face_n_hair",
    "b_upper_body": SORTED_OUTPUT_DIR / "b_upper_body",
    "c_full_body": SORTED_OUTPUT_DIR / "c_full_body"
}
# --- End Configuration ---

def gaussian_score(x, mu, sigma):
    """ (Unchanged) Applies a Gaussian bell curve. """
    return math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def get_source_video(filename):
    """ (Unchanged) Extracts source video name. """
    try:
        return filename.split("_frame_")[0]
    except IndexError:
        # Fallback for filenames that might not match
        return "unknown_source"

def get_frame_index(filename):
    """
    Extracts the frame index (e.g., 123) from a filename
    (e.g., "VideoName_frame_000123.jpg").
    """
    match = re.search(r'_frame_(\d+)\.(jpg|jpeg|png|webp)$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

def get_brightness(image_gray):
    """ Scores brightness, ideal is 128. """
    mean = np.mean(image_gray)
    score = 1.0 - (abs(mean - IDEAL_BRIGHTNESS) / IDEAL_BRIGHTNESS)
    return max(0, score)

def get_pose_score(pose_landmarks, category_name):
    """ Scores pose visibility based on category. """
    if not pose_landmarks: return 0.0
    landmarks = pose_landmarks.landmark
    def is_visible(landmark):
        return landmark.visibility > VISIBILITY_THRESHOLD

    if category_name == "a_face_n_hair":
        face_parts = [landmarks[p] for p in [PoseLandmark.NOSE, PoseLandmark.LEFT_EYE, PoseLandmark.RIGHT_EYE, PoseLandmark.MOUTH_LEFT, PoseLandmark.MOUTH_RIGHT]]
        score = sum(is_visible(part) for part in face_parts) / len(face_parts)
        if is_visible(landmarks[PoseLandmark.LEFT_ANKLE]) or is_visible(landmarks[PoseLandmark.RIGHT_ANKLE]):
            score *= 0.5 # Penalize face shots with feet visible
        return score
    elif category_name == "b_upper_body":
        upper_body_parts = [landmarks[p] for p in [PoseLandmark.NOSE, PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP]]
        score = sum(is_visible(part) for part in upper_body_parts) / len(upper_body_parts)
        if is_visible(landmarks[PoseLandmark.LEFT_ANKLE]) or is_visible(landmarks[PoseLandmark.RIGHT_ANKLE]):
            score *= 0.5 # Penalize upper body shots with feet visible
        return score
    elif category_name == "c_full_body":
        full_body_parts = [landmarks[p] for p in [PoseLandmark.NOSE, PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP, PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE]]
        score = sum(is_visible(part) for part in full_body_parts) / len(full_body_parts)
        return score
    return 0.0

def process_image_v4(img_path, image_rgb, image_gray, holistic, segmentation, pose, category_name):
    """
    Runs the full v4 scoring pipeline.
    Parses and returns 'frame_index'.
    """
    try:
        pose_results = pose.process(image_rgb)
        holistic_results = holistic.process(image_rgb)
        seg_results = segmentation.process(image_rgb)
    except Exception as e:
        print(f"Warning: MediaPipe error on {img_path.name}: {e}")
        return None

    S_brightness = get_brightness(image_gray)
    S_pose = get_pose_score(pose_results.pose_landmarks, category_name)
    
    raw_face_sharpness = 0.0
    raw_body_sharpness = 0.0
    focus_ratio = 1.0 

    try:
        laplacian_map = cv2.Laplacian(image_gray, cv2.CV_64F)
        if holistic_results.face_landmarks:
            face_landmarks = holistic_results.face_landmarks.landmark
            face_points = np.array([[int(p.x * image_rgb.shape[1]), int(p.y * image_rgb.shape[0])] for p in face_landmarks])
            
            face_mask = np.zeros(image_gray.shape, dtype=np.uint8)
            convexhull = cv2.convexHull(face_points)
            cv2.fillConvexPoly(face_mask, convexhull, 255)
            
            face_pixels = laplacian_map[face_mask > 0]
            if face_pixels.size > 0:
                raw_face_sharpness = np.var(face_pixels)

            person_mask_raw = seg_results.segmentation_mask
            person_mask = (person_mask_raw > 0.5).astype(np.uint8) * 255
            body_only_mask = cv2.subtract(person_mask, face_mask)
            
            body_pixels = laplacian_map[body_only_mask > 0]
            if body_pixels.size > 0:
                raw_body_sharpness = np.var(body_pixels)

            if raw_body_sharpness > 1e-6:
                focus_ratio = raw_face_sharpness / raw_body_sharpness
            elif raw_face_sharpness > 1e-6:
                focus_ratio = 100.0 # Face is sharp, body is not (good!)
    except Exception as e:
        pass # This can fail (e.g., no face), default scores are fine
        
    S_focus_score = gaussian_score(focus_ratio, GAUSS_MU, GAUSS_SIGMA)
    image_pil = Image.fromarray(image_rgb)
    phash = str(imagehash.phash(image_pil))

    return {
        "file": str(img_path.name),
        "path": str(img_path),
        "phash": phash,
        "source_video": get_source_video(img_path.name),
        "frame_index": get_frame_index(img_path.name),
        "scores": {
            "S_brightness": S_brightness, "S_pose": S_pose,
            "S_focus_score": S_focus_score, "raw_face_sharpness": raw_face_sharpness,
            "raw_body_sharpness": raw_body_sharpness, "focus_ratio": focus_ratio
        }
    }

def main():
    print("🚀 Starting v4 Scoring & v3+Temporal De-duplication Process...")
    all_final_results = {}

    print("Building video manifest (Pass 0)...")
    video_manifest = defaultdict(int)
    all_image_files = []
    
    for cat_path in CATEGORIES.values():
        if cat_path.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                 all_image_files.extend(list(cat_path.glob(ext)))
    
    if not all_image_files:
        print("No images found in any category. Cannot build manifest. Exiting.")
        return
        
    for img_path in tqdm(all_image_files, desc="Scanning frames"):
        source_video = get_source_video(img_path.name)
        frame_index = get_frame_index(img_path.name)
        if frame_index > video_manifest[source_video]:
            video_manifest[source_video] = frame_index
    
    print(f"Manifest built. Found {len(video_manifest)} source videos.")

    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose, \
         mp_holistic.Holistic(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as holistic, \
         mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as segmentation:
        
        for cat_name, cat_path in CATEGORIES.items():
            print(f"\nProcessing category: {cat_name}")
            if not cat_path.is_dir():
                print(f"Warning: Folder not found {cat_path}. Skipping.")
                continue

            image_files = list(cat_path.glob("*.jpg")) + list(cat_path.glob("*.jpeg")) + \
                          list(cat_path.glob("*.png")) + list(cat_path.glob("*.webp"))
            
            if not image_files:
                print("No images found in this category. Skipping.")
                continue

            all_category_scores = []
            skipped_count = 0
            for img_path in tqdm(image_files, desc=f"Scoring {cat_name} (Pass 1)"):
                try:
                    image = cv2.imread(str(img_path))
                    if image is None: 
                        skipped_count += 1; continue
                    
                    # --- MODIFICATION: The 1024x1024 gate is REMOVED ---
                    # h, w, _ = image.shape
                    # if h != 1024 or w != 1024:
                    #     skipped_count += 1; continue
                    # --- END MODIFICATION ---
                        
                    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    result_data = process_image_v4(img_path, image_rgb, image_gray, holistic, segmentation, pose, cat_name)
                    if result_data:
                        all_category_scores.append(result_data)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            if skipped_count > 0:
                # Clarified the message now that the 1024 gate is gone
                print(f"INFO: Skipped {skipped_count} corrupt images.")
            if not all_category_scores:
                print("No valid images scored in this category. Skipping.")
                continue

            print(f"Normalizing {len(all_category_scores)} images (Pass 2)...")
            # Find the max raw face sharpness for normalization
            max_face_sharpness = max(item['scores']['raw_face_sharpness'] for item in all_category_scores)
            if max_face_sharpness == 0: max_face_sharpness = 1.0 # Avoid div by zero

            for item in all_category_scores:
                S_abs_face_sharp = item['scores']['raw_face_sharpness'] / max_face_sharpness
                item['scores']['S_absolute_face_sharpness'] = S_abs_face_sharp
                
                item['final_score'] = (
                    item['scores']['S_pose'] * WEIGHTS['pose'] +
                    item['scores']['S_brightness'] * WEIGHTS['brightness'] +
                    item['scores']['S_absolute_face_sharpness'] * WEIGHTS['absolute_face_sharpness'] +
                    item['scores']['S_focus_score'] * WEIGHTS['focus_score']
                )

            all_category_scores.sort(key=lambda x: x["final_score"], reverse=True)
            
            print(f"Building unique top {TOP_N_IMAGES} list (3-Stage Filter)...")
            top_n_items = []
            seen_global_hashes = set()
            seen_source_hashes = defaultdict(list)
            seen_source_frame_indices = defaultdict(list)
            
            is_face_category = (cat_name == "a_face_n_hair")
            if is_face_category:
                print("INFO: Applying 2-stage pHash filter for 'face' category.")
            else:
                print("INFO: Applying 3-stage pHash + Temporal filter for 'body' category.")

            for item in tqdm(all_category_scores, desc=f"Filtering unique {cat_name}"):
                if len(top_n_items) >= TOP_N_IMAGES:
                    break
                current_hash = imagehash.hex_to_hash(item["phash"])
                current_source_video = item["source_video"]
                
                # Filter 1: Global (Strict)
                is_global_dupe = False
                for seen_hash in seen_global_hashes:
                    if (current_hash - seen_hash) < GLOBAL_HASH_THRESHOLD:
                        is_global_dupe = True; break
                if is_global_dupe: continue 

                # Filter 2: Intra-Source (Lax)
                is_source_dupe = False
                for seen_hash in seen_source_hashes[current_source_video]:
                    if (current_hash - seen_hash) < INTRA_SOURCE_HASH_THRESHOLD:
                        is_source_dupe = True; break
                if is_source_dupe: continue
                
                # Filter 3: Temporal Quota (for Body categories only)
                if not is_face_category:
                    selected_indices = seen_source_frame_indices[current_source_video]
                    selected_count = len(selected_indices)
                    
                    if selected_count == 5:
                        total_frames = video_manifest[current_source_video]
                        if total_frames == 0: continue # Cannot perform test, reject
                        avg_index = sum(selected_indices) / 5
                        midpoint = total_frames / 2
                        current_index = item["frame_index"]
                        test_passed = False
                        if avg_index < midpoint: # Avg is in 1st half
                            if current_index > (avg_index + total_frames) / 2: test_passed = True
                        else: # Avg is in 2nd half
                            if current_index < (avg_index / 2): test_passed = True
                        if not test_passed: continue # Reject: Failed temporal spread test
                    
                    elif selected_count >= 6:
                        continue # Reject: Hard cap of 6 images reached
                
                # --- ACCEPT ---
                top_n_items.append(item)
                seen_global_hashes.add(current_hash)
                seen_source_hashes[current_source_video].append(current_hash)
                if not is_face_category:
                    seen_source_frame_indices[current_source_video].append(item["frame_index"])
            
            all_final_results[cat_name] = top_n_items
            dest_folder = SORTED_OUTPUT_DIR / f"{cat_name}{OUTPUT_FOLDER_SUFFIX}"
            
            print(f"Copying {len(top_n_items)} unique top images to {dest_folder}...")
            if dest_folder.exists(): rmtree(dest_folder)
            dest_folder.mkdir(parents=True)
            
            for item in top_n_items:
                copy2(item["path"], dest_folder / item["file"])

    print(f"\nSaving detailed v3 scoring report to {OUTPUT_JSON_FILE}...")
    with open(OUTPUT_JSON_FILE, 'w') as f:
        json.dump(all_final_results, f, indent=2)

    print("\n--- v3 Process Complete ---")
    print("Scoring and ranking finished.")
    print(f"Top unique images saved to folders ending in '{OUTPUT_FOLDER_SUFFIX}'")

if __name__ == "__main__":
    main()
