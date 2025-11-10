Automated SDXL Dataset Curation Pipeline

This project is a complete, containerized pipeline for processing a large, raw pool of video frames (e.g., 200k+) into a small, high-quality, and varied dataset for AI model training (e.g., SDXL LoRA).

It automatically sifts, sorts, scores, and de-duplicates images to find the best available frames, solving common dataset problems like poor focus and low variety.

Key Features

Smarter Sorting (v2): Uses MediaPipe (Face Detection, Pose) to sort images into categories. It now includes a 30% Pose Area Gate to disqualify images with tiny, distant poses from the upper_body and full_body training pools, ensuring higher-quality inputs for the scoring script.

Advanced Focus Scoring (v4): Implements a "v4" scoring model that uses 3 separate ML models (Holistic, Selfie Segmentation, Pose) to find images where the subject's face is the sharpest object in the frame, penalizing images with blurry faces but sharp backgrounds.

3-Stage De-duplication Gauntlet (v3): Ensures a high-variety dataset by combining three filters:

Global pHash: A strict filter to remove near-identical twins.

Intra-Source pHash: A laxer filter to ensure visual variety from the same video.

Temporal Quota: A 4-frame hard cap for body categories that forces selections to be from different times in the video (e.g., 2 from the start, 2 from the end).

Automated Captioning: Includes scripts to auto-caption the final dataset using the Gemini or Vertex AI APIs.

Containerized & Portable: The entire environment (OpenCV, MediaPipe, PyTorch) is containerized with Docker, making it 100% reproducible.

The Pipeline: How It Works

The scripts are numbered in the order you should run them.

01_extract_frames.py:

Input: Raw video files in /data/source_videos.

Output: 1024x1024 center-cropped JPG frames in /data/source_images, organized into subfolders (e.g., 240,000+ images).

02_sort_dataset.py:

Input: The 240,000+ images in /data/source_images.

Output: Images are sorted by pose/shot type. This script now includes a 30% pose area gate. If a full_body or upper_body candidate's pose covers less than 30% of the image, it's disqualified into a f_needs_review_pose_too_small folder, keeping the training pools clean. Reports results to TensorBoard.

03_score_and_rank_v3.py:

This is the core logic.

Input: The clean sorted images from Step 2.

Process: Runs the v4 "Focus-Weighted" scoring and the "3-Stage" de-duplication filter.

a_face_n_hair uses a 2-stage pHash filter (which works well for faces).

b_upper_body & c_full_body use the 3-stage pHash + Temporal Quota filter to force timeline variety.

Output: The Top 60 best, most unique images for each category, saved to new folders (e.g., /data/sorted_output/a_face_n_hair_top60_v4_temporal).

04_caption_dataset_*.py:

Input: The final "Top 60" folders.

Process: Generates a descriptive caption for each image and saves it as a .txt file.

Output: A complete, captioned, high-quality dataset ready for AI training.

Setup & Usage

1. Initial Setup

Clone the repo:

git clone [https://github.com/your_username/your_repo_name.git](https://github.com/your_username/your_repo_name.git)
cd your_repo_name


Create data structure: This project uses a /data directory (which is git-ignored) to hold all images and videos.

mkdir -p data/source_videos
mkdir -p data/source_images
mkdir -p data/sorted_output
mkdir -p data/tensorboard_logs


Add your secrets:

Add your Google service-account-key.json file to the root of the project.

Never commit this file. The .gitignore should prevent this.

You can use service-account-key.json.example as a template.

Add your videos:

Place your .mp4, .mov, etc. files into the /data/source_videos folder.

2. Running the Pipeline

Build and start the container: This will build the Docker image and start the utils (Python) and tensorboard services.

docker compose up -d --build


Open TensorBoard: You can now view the sorting logs by navigating to http://localhost:6006 in your browser.

Enter the container:

docker exec -it utils_service /bin/bash


(Your prompt will change, you are now inside the container at the /projects directory).

Run the scripts in order:

# Step 1: Extract frames (This will take a while)
python3 01_extract_frames.py

# Step 2: Sort all frames (This will also take a while)
# This v2 sorter uses the 30% pose area gate.
python3 02_sort_dataset.py

# Step 3: Score and rank to find the Top 60 (The "magic" step)
python3 03_score_and_rank_v3.py

# Step 4: Caption your final dataset
# Edit the script first to set your API key and target folder
nano 04a_caption_gemini_api.py
python3 04a_caption_gemini_api.py
