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

# --- Configuration ---

# 1. MediaPipe Models (NEW: Added Holistic and Selfie Segmentation)
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_selfie_segmentation = mp.solutions.selfie_segmentation
PoseLandmark = mp_pose.PoseLandmark

# 2. New v4 Weights (Heavily biased toward new focus metrics)
WEIGHTS = {
    "pose": 0.20,
    "brightness": 0.10,
    "absolute_face_sharpness": 0.35,
    "focus_score": 0.35
}

# 3. New De-duplication Filters
TOP_N_IMAGES = 60
GLOBAL_HASH_THRESHOLD = 3      # WAS: 6. Now very strict.
INTRA_SOURCE_HASH_THRESHOLD = 10 # NEW: More-lax filter for same-video variety.

# 4. Scoring "Goldilocks" values
IDEAL_BRIGHTNESS = 128.0
VISIBILITY_THRESHOLD = 0.5
GAUSS_MU = 1.0   # Peak of the bell curve (ideal ratio)
GAUSS_SIGMA = 0.25 # Steepness of the bell curve (tune this to adjust penalty)

# 5. Paths (Unchanged from your v1 script)
BASE_PROJECT_DIR = Path("/projects")
SORTED_OUTPUT_DIR = BASE_PROJECT_DIR / "sorted_output"
OUTPUT_JSON_FILE = BASE_PROJECT_DIR / "scoring_results_v2.json" # New JSON name

# --- CRITICAL: Matched to your sort_dataset.py folder names ---
CATEGORIES = {
    "a_face_n_hair": SORTED_OUTPUT_DIR / "a_face_n_hair",
    "b_upper_body": SORTED_OUTPUT_DIR / "b_upper_body",
    "c_full_body": SORTED_OUTPUT_DIR / "c_full_body"
}

# --- New Helper Functions ---

def gaussian_score(x, mu, sigma):
    """
    Applies a Gaussian (bell curve) function to the FocusRatio.
    Score is 1.0 at the peak (mu) and drops off.
    """
    return math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def get_source_video(filename):
    """
    Extracts the source video name (e.g., "VideoA") from a frame
    filename (e.g., "VideoA_frame_0001.jpg").
    This relies on the format from extract_frames_1024.py
    """
    # This split is reliable based on your extract_frames_1024.py
    return filename.split("_frame_")[0]

# --- Scoring Functions (v1 functions updated for v4) ---

def get_brightness(image_gray):
    """ (Unchanged from v1) Scores brightness, ideal is 128. """
    mean = np.mean(image_gray)
    score = 1.0 - (abs(mean - IDEAL_BRIGHTNESS) / IDEAL_BRIGHTNESS)
    return max(0, score)

def get_pose_score(pose_landmarks, category_name):
    """ (Unchanged from v1) Scores pose visibility based on category. """
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
    Runs the full v4 scoring pipeline on a single image.
    Returns a dictionary of all raw scores.
    """
    # 1. Run all 3 MediaPipe models
    try:
        pose_results = pose.process(image_rgb)
        holistic_results = holistic.process(image_rgb)
        seg_results = segmentation.process(image_rgb)
    except Exception as e:
        print(f"Warning: MediaPipe error on {img_path.name}: {e}")
        return None

    # 2. Get standard scores (v1)
    S_brightness = get_brightness(image_gray)
    S_pose = get_pose_score(pose_results.pose_landmarks, category_name)
    
    # --- 3. Start New v4 Focus-Weighted Scoring ---
    raw_face_sharpness = 0.0
    raw_body_sharpness = 0.0
    focus_ratio = 1.0 # Default to 1.0 (perfect) if data is missing

    try:
        # Get global Laplacian "edge map"
        laplacian_map = cv2.Laplacian(image_gray, cv2.CV_64F)

        # 3a. Create Face Mask
        if holistic_results.face_landmarks:
            face_landmarks = holistic_results.face_landmarks.landmark
            face_points = np.array([[int(p.x * image_rgb.shape[1]), int(p.y * image_rgb.shape[0])] for p in face_landmarks])
            
            # Create convex hull for the face mask
            face_mask = np.zeros(image_gray.shape, dtype=np.uint8)
            convexhull = cv2.convexHull(face_points)
            cv2.fillConvexPoly(face_mask, convexhull, 255)
            
            # Get sharpness of just the face pixels
            face_pixels = laplacian_map[face_mask > 0]
            if face_pixels.size > 0:
                raw_face_sharpness = np.var(face_pixels)

            # 3b. Create Body-Only Mask
            # Get the full person mask from SelfieSegmentation
            person_mask_raw = seg_results.segmentation_mask
            person_mask = (person_mask_raw > 0.5).astype(np.uint8) * 255
            
            # Subtract the face mask to get a "body-only" mask (hair, clothes, etc)
            body_only_mask = cv2.subtract(person_mask, face_mask)
            
            # Get sharpness of just the body-only pixels
            body_pixels = laplacian_map[body_only_mask > 0]
            if body_pixels.size > 0:
                raw_body_sharpness = np.var(body_pixels)

            # 3c. Calculate Focus Ratio (Handle potential division by zero)
            if raw_body_sharpness > 1e-6:
                focus_ratio = raw_face_sharpness / raw_body_sharpness
            elif raw_face_sharpness > 1e-6:
                focus_ratio = 100.0 # Face is sharp, body is not (good!)
            # Else, both are 0, ratio stays 1.0 (neutral)

    except Exception as e:
        # This can fail (e.g., no face/body found), default scores are fine
        pass 
        
    # 4. Calculate Final v4 Scores
    S_focus_score = gaussian_score(focus_ratio, GAUSS_MU, GAUSS_SIGMA)
    
    # Get pHash
    image_pil = Image.fromarray(image_rgb)
    phash = str(imagehash.phash(image_pil))

    # Return all metrics. Normalization happens in main loop.
    return {
        "file": str(img_path.name),
        "path": str(img_path),
        "phash": phash,
        "source_video": get_source_video(img_path.name),
        "scores": {
            "S_brightness": S_brightness,
            "S_pose": S_pose,
            "S_focus_score": S_focus_score,
            "raw_face_sharpness": raw_face_sharpness,
            "raw_body_sharpness": raw_body_sharpness,
            "focus_ratio": focus_ratio
        }
    }

# --- Main Execution ---
def main():
    print("🚀 Starting v4 Scoring & v3 De-duplication Process...")
    all_final_results = {}
    
    # Initialize all 3 MP models. model_complexity=1 is a good balance.
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

            # --- Pass 1: Gather all scores ---
            all_category_scores = []
            skipped_count = 0
            for img_path in tqdm(image_files, desc=f"Scoring {cat_name} (Pass 1)"):
                try:
                    image = cv2.imread(str(img_path))
                    if image is None: 
                        skipped_count += 1
                        continue
                    
                    # Gate: All images must be 1024x1024
                    h, w, _ = image.shape
                    if h != 1024 or w != 1024:
                        skipped_count += 1
                        continue
                        
                    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Run the full v4 scoring pipeline
                    result_data = process_image_v4(img_path, image_rgb, image_gray, holistic, segmentation, pose, cat_name)
                    
                    if result_data:
                        all_category_scores.append(result_data)
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            if skipped_count > 0:
                print(f"INFO: Skipped {skipped_count} images (corrupt or not 1024x1024).")
            
            if not all_category_scores:
                print("No valid images scored in this category. Skipping.")
                continue

            # --- Pass 2: Normalize and Calculate Final Score ---
            print(f"Normalizing {len(all_category_scores)} images (Pass 2)...")
            
            # Find the max raw face sharpness for normalization
            max_face_sharpness = max(item['scores']['raw_face_sharpness'] for item in all_category_scores)
            if max_face_sharpness == 0: max_face_sharpness = 1.0 # Avoid div by zero

            for item in all_category_scores:
                # Normalize AbsoluteFaceSharpness from 0.0 to 1.0
                S_abs_face_sharp = item['scores']['raw_face_sharpness'] / max_face_sharpness
                item['scores']['S_absolute_face_sharpness'] = S_abs_face_sharp
                
                # Calculate weighted FinalScore
                item['final_score'] = (
                    item['scores']['S_pose'] * WEIGHTS['pose'] +
                    item['scores']['S_brightness'] * WEIGHTS['brightness'] +
                    item['scores']['S_absolute_face_sharpness'] * WEIGHTS['absolute_face_sharpness'] +
                    item['scores']['S_focus_score'] * WEIGHTS['focus_score']
                )

            # Sort by final_score (highest first)
            all_category_scores.sort(key=lambda x: x["final_score"], reverse=True)
            
            # --- Pass 3: New v3 Two-Stage De-duplication Filter ---
            print(f"Building unique top {TOP_N_IMAGES} list (v3 de-duplication)...")
            top_n_items = []
            seen_global_hashes = set()             # Filter 1: Strict global filter
            seen_source_hashes = defaultdict(list) # Filter 2: Lax intra-source filter

            for item in tqdm(all_category_scores, desc=f"Finding unique {cat_name}"):
                if len(top_n_items) >= TOP_N_IMAGES:
                    break # We found our Top 60

                current_hash = imagehash.hex_to_hash(item["phash"])
                
                # --- Filter 1: Global (Strict) ---
                is_global_dupe = False
                for seen_hash in seen_global_hashes:
                    if (current_hash - seen_hash) < GLOBAL_HASH_THRESHOLD:
                        is_global_dupe = True
                        break
                if is_global_dupe:
                    continue # Reject: Too similar to *any* other selected image

                # --- Filter 2: Intra-Source (Lax) ---
                source_video = item["source_video"]
                is_source_dupe = False
                for seen_hash in seen_source_hashes[source_video]:
                    if (current_hash - seen_hash) < INTRA_SOURCE_HASH_THRESHOLD:
                        is_source_dupe = True
                        break
                if is_source_dupe:
                    continue # Reject: Not different enough from *its own video*
                
                # --- ACCEPT ---
                # Image passed both filters!
                top_n_items.append(item)
                seen_global_hashes.add(current_hash)
                seen_source_hashes[source_video].append(current_hash)
            
            all_final_results[cat_name] = top_n_items
            
            # --- Copy Top N Images ---
            dest_folder = SORTED_OUTPUT_DIR / f"{cat_name}_top{TOP_N_IMAGES}_v4" # New folder name
            
            print(f"Copying {len(top_n_items)} unique top images to {dest_folder}...")
            if dest_folder.exists():
                rmtree(dest_folder)
            dest_folder.mkdir(parents=True)
            
            for item in top_n_items:
                copy2(item["path"], dest_folder / item["file"])

    # --- Save JSON Report ---
    print(f"\nSaving detailed v4 scoring report to {OUTPUT_JSON_FILE}...")
    with open(OUTPUT_JSON_FILE, 'w') as f:
        json.dump(all_final_results, f, indent=2)

    print("\n--- v4 Process Complete ---")
    print("Scoring and ranking finished.")
    print("Top unique, focus-weighted images for each category have been copied.")

if __name__ == "__main__":
    main()
