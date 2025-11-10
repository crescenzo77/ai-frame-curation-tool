import os
import shutil
import cv2
import mediapipe as mp
from tqdm import tqdm
import warnings
import logging
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path  # --- MODIFIED ---: Import Pathlib

# --- Configuration ---
# All paths are relative to the '/projects' mount
# from the docker-compose.yml
SOURCE_DIR = "/projects/source_images"
OUTPUT_DIR = "/projects/sorted_output"
TENSORBOARD_DIR = "/projects/tensorboard_logs"

# Heuristics (you can tune these)
FACE_BBOX_AREA_THRESHOLD = 0.20  # 20% of image area = close-up
VISIBILITY_THRESHOLD = 0.5      # How confident MediaPipe must be of a landmark
IMAGE_SAMPLE_COUNT = 10         # Log first 10 images of each class to TB
# --- End Configuration ---

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress TensorFlow/MediaPipe warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Define folder structure
FOLDERS = {
    "face": "a_face_n_hair",                 # --- MODIFIED ---: Matched score_and_rank.py
    "upper_body": "b_upper_body",           # --- MODIFIED ---: Matched score_and_rank.py
    "full_body": "c_full_body",             # --- MODIFIED ---: Matched score_and_rank.py
    "multi_person": "d_needs_review_multi_person", # --- MODIFIED ---
    "no_person": "e_needs_review_no_person"        # --- MODIFIED ---
}

def setup_directories():
    """Creates the output directory structure."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Ensuring output directory exists: {OUTPUT_DIR}")
    for folder_name in FOLDERS.values():
        os.makedirs(os.path.join(OUTPUT_DIR, folder_name), exist_ok=True)
    
    # Also ensure TensorBoard dir exists
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    logging.info(f"TensorBoard log directory: {TENSORBOARD_DIR}")

def sort_images():
    """Main function to sort images using MediaPipe."""
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(TENSORBOARD_DIR)
    category_counts = {name: 0 for name in FOLDERS.values()}
    
    # Initialize MediaPipe models
    mp_face_detection = mp.solutions.face_detection
    mp_pose = mp.solutions.pose

    # Use 'with' statements for safe resource handling
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection, \
         mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        # --- MODIFIED ---: Use Path.rglob() to search recursively
        logging.info(f"Scanning for images recursively in {SOURCE_DIR}...")
        source_path = Path(SOURCE_DIR)
        
        if not source_path.is_dir():
            logging.error(f"FATAL: Source directory not found: {SOURCE_DIR}")
            writer.close()
            return
            
        image_files = list(source_path.rglob("*.jpg")) + \
                      list(source_path.rglob("*.jpeg")) + \
                      list(source_path.rglob("*.png")) + \
                      list(source_path.rglob("*.webp"))
        # --- END MODIFIED ---
        
        if not image_files:
            logging.warning(f"No images found in {SOURCE_DIR}. Exiting.")
            writer.close()
            return

        logging.info(f"Found {len(image_files)} images. Starting analysis...")

        # --- MODIFIED ---: Loop over Path objects, not just filenames
        for image_path in tqdm(image_files, desc="Sorting Images"):
            
            image_name = image_path.name # Get just the filename
            
            try:
                # cv2.imread needs a string path
                image = cv2.imread(str(image_path)) 
                if image is None:
                    logging.warning(f"Could not read {image_name}. Skipping.")
                    continue
                
                h, w, _ = image.shape
                image_area = h * w
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # --- 1. Face Detection ---
                face_results = face_detection.process(image_rgb)
                
                if not face_results.detections:
                    dest_folder = FOLDERS["no_person"]
                elif len(face_results.detections) > 1:
                    dest_folder = FOLDERS["multi_person"]
                else:
                    # --- 2. Close-up Check (One face detected) ---
                    detection = face_results.detections[0]
                    bbox = detection.location_data.relative_bounding_box
                    face_bbox_area = (bbox.width * w) * (bbox.height * h)
                    face_ratio = face_bbox_area / image_area

                    if face_ratio > FACE_BBOX_AREA_THRESHOLD:
                        dest_folder = FOLDERS["face"]
                    else:
                        # --- 3. Pose Detection (Not a close-up, check pose) ---
                        pose_results = pose.process(image_rgb)
                        
                        if not pose_results.pose_landmarks:
                            dest_folder = FOLDERS["face"] # Has a face but no pose, default to face
                        else:
                            landmarks = pose_results.pose_landmarks.landmark
# --- REPLACE IT WITH THIS ---
                            shoulder_visible = (
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > VISIBILITY_THRESHOLD and
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > VISIBILITY_THRESHOLD
                            )
                            hip_visible = (
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > VISIBILITY_THRESHOLD and
                                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > VISIBILITY_THRESHOLD
                            )
                            ankle_visible = (
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility > VISIBILITY_THRESHOLD and
                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility > VISIBILITY_THRESHOLD
                            )

                            # --- 4. Categorize by Pose ---
                            if shoulder_visible and hip_visible and ankle_visible:
                                dest_folder = FOLDERS["full_body"]
                            elif shoulder_visible and hip_visible:
                                dest_folder = FOLDERS["upper_body"]
                            elif shoulder_visible:
                                dest_folder = FOLDERS["upper_body"]
                            else:
                                dest_folder = FOLDERS["face"]
                
                # --- 5. Copy file and log to TensorBoard ---
                
                # Copy the file (use str() for os.path.join)
                shutil.copy(str(image_path), os.path.join(OUTPUT_DIR, dest_folder, image_name))
                
                # Update counters
                category_counts[dest_folder] += 1
                
                # Log a sample to TensorBoard
                count = category_counts[dest_folder]
                if count <= IMAGE_SAMPLE_COUNT:
                    # Convert (H, W, C) to (C, H, W) for TensorBoard
                    image_rgb_chw = image_rgb.transpose((2, 0, 1))
                    writer.add_image(f"{dest_folder}/{image_name}", image_rgb_chw, global_step=count)

            except Exception as e:
                # Use image_name (filename) for logging error
                logging.error(f"Error processing {image_name}: {e}")
                continue
        # --- END MODIFIED LOOP ---
            
        # --- 6. Finalize TensorBoard Logging ---
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
    logging.info("\nBatch sorting complete!")
    logging.info(f"Sorted images are available in {OUTPUT_DIR}")
    logging.info(f"TensorBoard logs are available in {TENSORBOARD_DIR}")
