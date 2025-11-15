# AI LoRA Data Curation Pipeline

This project provides a complete, 6-step data curation pipeline to process a large library of source videos into a small, high-quality, LoRA-ready dataset.

It uses a "Vertical Slice Test" methodology (running on a `data_TEST` folder) to validate the entire pipeline before committing to a full production run.

## Features

* **Step 1:** Video Frame Extraction
* **Step 2:** Smart Culling with **YOLO-Pose** (and QA reports for rejects)
* **Step 3:** Subject Masking with **rembg**
* **Step 4:** Mask-Aware Scoring (filters for sharpness, blurriness, and mask quality)
* **Step 5:** Automated Background Replacement (pastes subject onto blurred, random backgrounds)
* **Step 6:** Automated Captioning with **Google Vertex AI** (Gemini)

## Requirements

* An NVIDIA GPU with CUDA
* Docker and Docker Compose
* A Google Cloud Project with the **Vertex AI API** enabled.
* A Google Cloud Service Account with the **"Vertex AI User"** role.

## Project Setup

1.  **Clone this repository.**

2.  **Add Your Data (in the `data/` folder):**
    The `data` folder is the main workspace and is ignored by Git. You must create and populate:
    * `./data/source_videos/`: Add all your `.mp4` source videos here.
    * `./data/00_background_library/`: Add all your `.jpg` background images here.
    * `./data/yolov8l-pose.pt`: Download the YOLOv8 Pose model and place the `.pt` file here.
    * `./data/data_TEST/`: Create this folder and mirror the structure above for your Vertical Slice Test.

3.  **Add Your Google Key:**
    * Download the JSON key for your Service Account.
    * Rename it to `service-account-key.json`.
    * Place it inside the `./docker/` folder. The `.gitignore` file will prevent it from being uploaded.

4.  **Configure Scripts:**
    * Open `scripts/05_caption.py` and edit the `default="your trigger, ",` line to your desired trigger words.

5.  **Build the Container:**
    ```bash
    # From the sdxl-dataset-pipeline root folder:
    docker compose -f ./docker/docker-compose.yml up -d --build
    ```

## Workflow: The 6-Step Pipeline

Enter your running container to perform all work.

```bash
docker compose -f ./docker/docker-compose.yml exec lora_utils /bin/bash
