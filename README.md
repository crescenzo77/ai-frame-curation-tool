# Automated SDXL Dataset Curation Pipeline

This project is a complete, containerized pipeline for processing a large, raw pool of video frames (e.g., 200k+) into a small, high-quality, and varied dataset for AI model training (e.g., SDXL LoRA).

It automatically sifts, sorts, scores, and de-duplicates images to find the *best available* frames, solving common dataset problems like poor focus and low variety.

## 🌟 Key Features

* **Automated Sorting:** Uses MediaPipe (Face Detection, Pose) to sort 1000s of images into categories like `face`, `upper_body`, and `full_body` in minutes.
* **Advanced Focus Scoring:** Implements a "v4" scoring model that uses 3 separate ML models (Holistic, Selfie Segmentation, Pose) to find images where the *subject's face is the sharpest object in the frame*, penalizing images with blurry faces but sharp backgrounds.
* **Dynamic De-duplication:** A "v3" two-stage pHash filter ensures a high-variety final dataset. It uses a strict *global* filter (to remove near-identical images) and a separate *intra-source* filter (to select visually distinct frames from the same video).
* **Automated Captioning:** Includes scripts to auto-caption the final dataset using the Gemini or Vertex AI APIs.
* **Containerized & Portable:** The entire environment (OpenCV, MediaPipe, PyTorch) is containerized with Docker, making it 100% reproducible.

## 📦 The Pipeline: How It Works

The scripts are numbered in the order you should run them.

1.  **`01_extract_frames.py`**:
    * **Input:** Raw video files in `/data/source_videos`.
    * **Output:** 1024x1024 center-cropped JPG frames in `/data/source_images`, organized into subfolders (e.g., 240,000+ images).

2.  **`02_sort_dataset.py`**:
    * **Input:** The 240,000+ images in `/data/source_images`.
    * **Output:** Images are sorted by pose/shot type into `/data/sorted_output` (e.g., `a_face_n_hair`, `b_upper_body`). Reports results to TensorBoard.

3.  **`03_score_and_rank_v2.py`**:
    * **This is the core logic.**
    * **Input:** The sorted images (e.g., 16,000 images in `a_face_n_hair`).
    * **Process:** Runs the v4 "Focus-Weighted" scoring and v3 "Dynamic De-duplication" filter on all candidates.
    * **Output:** The **Top 60** best, most unique images for each category, saved to new folders (e.g., `/data/sorted_output/a_face_n_hair_top60_v4`).

4.  **`04_caption_dataset_*.py`**:
    * **Input:** The final "Top 60" folders.
    * **Process:** Generates a descriptive caption for each image and saves it as a `.txt` file.
    * **Output:** A complete, captioned, high-quality dataset ready for AI training.

## 🛠️ Setup & Usage

### 1. Initial Setup

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/your_username/your_repo_name.git](https://github.com/your_username/your_repo_name.git)
    cd your_repo_name
    ```

2.  **Create `data` structure:** This project uses a `/data` directory (which is git-ignored) to hold all images and videos.
    ```bash
    mkdir -p data/source_videos
    mkdir -p data/source_images
    mkdir -p data/sorted_output
    mkdir -p data/tensorboard_logs
    ```

3.  **Add your secrets:**
    * Add your Google `service-account-key.json` file to the root of the project.
    * *Never* commit this file. The `.gitignore` should prevent this.
    * You can use `service-account-key.json.example` as a template.

4.  **Add your videos:**
    * Place your `.mp4`, `.mov`, etc. files into the `/data/source_videos` folder.

### 2. Running the Pipeline

1.  **Build and start the container:** This will build the Docker image and start the `utils` (Python) and `tensorboard` services.
    ```bash
    docker compose up -d --build
    ```

2.  **Open TensorBoard:** You can now view the sorting logs by navigating to `http://localhost:6006` in your browser.

3.  **Enter the container:**
    ```bash
    docker exec -it utils_service /bin/bash
    ```
    (Your prompt will change, you are now inside the container at the `/projects` directory).

4.  **Run the scripts *in order*:**

    ```bash
    # Step 1: Extract frames (This will take a while)
    python3 01_extract_frames.py

    # Step 2: Sort all frames (This will also take a while)
    python3 02_sort_dataset.py

    # Step 3: Score and rank to find the Top 60 (The "magic" step)
    python3 03_score_and_rank_v2.py

    # Step 4: Caption your final dataset
    # Edit the script first to set your API key and target folder
    nano 04a_caption_gemini_api.py
    python3 04a_caption_gemini_api.py
    ```
