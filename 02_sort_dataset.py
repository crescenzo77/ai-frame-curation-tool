#!/usr/bin/env python3

import os
import shutil
import cv2
import mediapipe as mp
from tqdm import tqdm
import warnings
import logging
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# --- Configuration ---
SOURCE_DIR = "/projects/source_images"
OUTPUT_DIR = "/projects/sorted_output"
TENSORBOARD_DIR = "/projects/tensorboard_logs"

FACE_BBOX_AREA_THRESHOLD = 0.20  # 20% of image area = close-up
VISIBILITY_THRESHOLD = 0.5     # How confident MediaPipe must be

# --- MODIFICATION: Set to a very permissive 2% ---
MIN_POSE_AREA_PERCENT = 2.0   # NOW 2% (was 10.0)

IMAGE_SAMPLE_COUNT = 10        
# --- End Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

FOLDERS = {
    "face": "a_face_n_hair",
    "upper_body": "b_upper_body",
    "full_body": "c_full_body",
    "multi_person": "d_needs_review_multi_person",
    "no_person": "e_needs_review_no_person",
    "pose_too_small": "f_needs_review_pose_too_small"
}

# (The get_pose_bbox_area_percent function is unchanged)
def get_pose_bbox_area_percent(pose_landmarks, h, w):
    if not pose_landmarks:
        return 0.0
    total_image_area = h * w
    if total_image_area == 0:
        return 0.0
    min_x, min_y = w, h
    max_x, max_y = 0, 0
    visible_landmarks_found = False
    for landmark in pose_landmarks.landmark:
        if landmark.visibility < VISIBILITY_THRESHOLD:
            continue
        visible_landmarks_found = True
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        min_x = min(min_x, x); min_y = min(min_y, y)
        max_x = max(max_x, x); max_y = max(max_y, y)
    if not visible_landmarks_found:
        return 0.0
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    bbox_area = bbox_width * bbox_height
    if bbox_width <= 0 or bbox_height <= 0:
        return 0.0
    return (bbox_area / total_image_area) * 100

def setup_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Ensuring output directory exists: {OUTPUT_DIR}")
    for folder_name in FOLDERS.values():
        os.makedirs(os.path.join(OUTPUT_DIR, folder_name), exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    logging.info(f"TensorBoard log directory: {TENSORBOARD_DIR}")

def sort_images():
    writer = SummaryWriter(TENSORBOARD_DIR)
    category_counts = {name: 0 for name in FOLDERS.values()}
    
    mp_face_detection = mp.solutions.face_detection
    mp_pose = mp.solutions.pose

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection, \
         mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        logging.info(f"Scanning for images recursively in {SOURCE_DIR}...")
        source_path = Path(SOURCE_DIR)
        
        if not source_path.is_dir():
            logging.error(f"FATAL: Source directory not found: {SOURCE_DIR}")
            writer.close(); return
            
        image_files = list(source_path.rglob("*.jpg")) + \
                      list(source_path.rglob("*.jpeg")) + \
                      list(source_path.rglob("*.png")) + \
                      list(source_path.rglob("*.webp"))
        
        if not image_files:
            logging.warning(f"No images found in {SOURCE_DIR}. Exiting."); 
            writer.close(); return

        logging.info(f"Found {len(image_files)} images. Starting v3-FINAL analysis (2% gate)...")

        for image_path in tqdm(image_files, desc="Sorting Images (v3-FINAL)"):
            image_name = image_path.name
            try:
                image = cv2.imread(str(image_path)) 
                if image is None:
                    logging.warning(f"Could not read {image_name}. Skipping."); continue
                
                h, w, _ = image.shape
                image_area = h * w
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                face_results = face_detection.process(image_rgb)
                
                if face_results.detections:
                    if len(face_results.detections) > 1:
                        dest_folder = FOLDERS["multi_person"]
                    else:
                        detection = face_results.detections[0]
                        bbox = detection.location_data.relative_bounding_box
                        face_bbox_area = (bbox.width * w) * (bbox.height * h)
                        face_ratio = face_bbox_area / image_area

                        if face_ratio > FACE_BBOX_AREA_THRESHOLD:
                            dest_folder = FOLDERS["face"]
                        else:
                            pose_results = pose.process(image_rgb)
                            if not pose_results.pose_landmarks:
                                dest_folder = FOLDERS["face"]
                            else:
                                landmarks = pose_results.pose_landmarks.landmark
                                shoulder_visible = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > VISIBILITY_THRESHOLD or landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > VISIBILITY_THRESHOLD)
                                hip_visible = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > VISIBILITY_THRESHOLD or landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > VISIBILITY_THRESHOLD)
                                ankle_visible = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility > VISIBILITY_THRESHOLD or landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility > VISIBILITY_THRESHOLD)

                                if shoulder_visible and hip_visible and ankle_visible:
                                    candidate_category = FOLDERS["full_body"]
                                elif shoulder_visible and hip_visible:
                                    candidate_category = FOLDERS["upper_body"]
                                elif shoulder_visible:
                                    candidate_category = FOLDERS["upper_body"]
                                else:
                                    candidate_category = FOLDERS["face"]
                                
                                if (candidate_category == FOLDERS["full_body"] or candidate_category == FOLDERS["upper_body"]):
                                    pose_area_percent = get_pose_bbox_area_percent(pose_results.pose_landmarks, h, w)
                                    if pose_area_percent < MIN_POSE_AREA_PERCENT: # Now 2.0
                                        dest_folder = FOLDERS["pose_too_small"]
                                    else:
                                        dest_folder = candidate_category
                                else:
                                    dest_folder = candidate_category
                else:
                    # Pose Fallback Logic
                    pose_results = pose.process(image_rgb)
                    if not pose_results.pose_landmarks:
                        dest_folder = FOLDERS["no_person"]
                    else:
                        landmarks = pose_results.pose_landmarks.landmark
                        shoulder_visible = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > VISIBILITY_THRESHOLD or landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > VISIBILITY_THRESHOLD)
                        hip_visible = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > VISIBILITY_THRESHOLD or landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > VISIBILITY_THRESHOLD)
                        ankle_visible = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility > VISIBILITY_THRESHOLD or landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility > VISIBILITY_THRESHOLD)

                        if shoulder_visible and hip_visible and ankle_visible:
                            candidate_category = FOLDERS["full_body"]
                        elif shoulder_visible and hip_visible:
                            candidate_category = FOLDERS["upper_body"]
                        elif shoulder_visible:
                            candidate_category = FOLDERS["upper_body"]
                        else:
                            dest_folder = FOLDERS["no_person"]
                            candidate_category = None

                        if candidate_category:
                            pose_area_percent = get_pose_bbox_area_percent(pose_results.pose_landmarks, h, w)
                            if pose_area_percent < MIN_POSE_AREA_PERCENT: # Now 2.0
                                dest_folder = FOLDERS["pose_too_small"]
                            else:
                                dest_folder = candidate_category
                
                shutil.copy(str(image_path), os.path.join(OUTPUT_DIR, dest_folder, image_name))
                category_counts[dest_folder] += 1
                
                count = category_counts[dest_folder]
                if count <= IMAGE_SAMPLE_COUNT:
                    image_rgb_chw = image_rgb.transpose((2, 0, 1))
                    writer.add_image(f"{dest_folder}/{image_name}", image_rgb_chw, global_step=count)
            except Exception as e:
                logging.error(f"Error processing {image_name}: {e}")
                continue
            
    logging.info("Logging final counts to TensorBoard...")
    total_processed = 0
    for category, count in category_counts.items():
        writer.add_scalar(f"Count/{category}", count, global_step=0)
        total_processed += count
    
    writer.add_scalar("Count/TotalProcessed", total_processed, global_step=0)
    writer.close()

if __name__ == "__main__":
    setup_directories()
    sort_images()
    logging.info(f"\nBatch sorting complete! (v3-FINAL with 2% pose gate & pose-fallback)")
    logging.info(f"Sorted images are available in {OUTPUT_DIR}")
    logging.info(f"TensorBoard logs are available in {TENSORBOARD_DIR}")
