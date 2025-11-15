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
    * `./data/data_TEST/`: (Recommended) Create this folder and mirror the structure above for your Vertical Slice Test.

3.  **Add Your Google Key:**
    * Download the JSON key for your Service Account.
    * Rename it to `service-account-key.json`.
    * Place it inside the `./docker/` folder. The `.gitignore` file will prevent it from being uploaded.

4.  **Configure Scripts:**
    * Open `scripts/05_caption.py` and edit the `default="your trigger, ",` line to your desired trigger words.
    * Review the filter settings at the top of `scripts/04_score_masked.py` (e.g., `MIN_SHARPNESS_THRESHOLD`).

5.  **Build the Container:**
    ```bash
    # From the sdxl-dataset-pipeline root folder:
    docker compose -f ./docker/docker-compose.yml up -d --build
    ```

## Workflow: The 6-Step Pipeline & QA

Enter your running container to perform all work.

```bash
docker compose -f ./docker/docker-compose.yml exec lora_utils /bin/bash
Inside the container, your data is at /projects and your scripts are at /scripts. Run the scripts in order.

(Note: These commands are for the full run (--base_dir /projects). For testing, use --base_dir /projects/data_TEST)

Step 1: Extract Frames
This pipeline is designed to run on a single Linux machine. The first step is to get your video frames into the /projects/01_source_frames directory (which maps to ./data/01_source_frames on your host).

You can use any tool you prefer for this, such as ffmpeg.

Bash

# Example ffmpeg command to extract 1 frame per second
# (Run this inside the container to see the /projects paths)
ffmpeg -i /projects/source_videos/my_video.mp4 -r 1 /projects/01_source_frames/my_video/frame_%06d.jpg
(Advanced Note: For multi-machine workflows, you can extract frames on a separate computer (e.g., a Mac with a fast media engine) and save them to the data/01_source_frames folder via a network share, as long as the files are present before you run Step 2.)

Step 2: Sort Candidates
This culls all raw frames and sorts the "good" ones into categories.

Bash

python3 /scripts/02_sort.py --base_dir /projects
🔬 QA Control (Step 2)
Run the reject report script to find "false negatives."

Bash

python3 /scripts/02b_qa_rejects.py --base_dir /projects
What to look for: Open the generated 02b_reject_qa_report.html.

The "Random Sample" section gives you a gut check. Is it mostly junk? Good.

The "Top 100 Sharpest" section is the real test. Are these sharp, high-quality images of your subject? If yes, your 02_sort.py script's CONF_THRESHOLD is too high, and you are accidentally rejecting good data.

Step 3: Mask Subjects
This runs rembg on all sorted candidates from Step 2.

Bash

python3 /scripts/03_mask_subjects.py --base_dir /projects
🔬 QA Control (Step 3)
Run the mask-viewing report.

Bash

python3 /scripts/03b_qa_masking.py --base_dir /projects
What to look for: Open the 03b_mask_qa_report.html. This report shows a side-by-side of the original vs. the masked image.

Scan the images. Are the masks generally high quality? Are "bites" (solidity errors) being taken out of the subject? Are extra background objects (contour errors) being included? This visual check helps you decide if the mask filters in the next step are necessary.

Step 4: Score Images
This is the great filter. It discards blurry, low-quality, or bad-mask images based on the hard-coded filters in the script.

Bash

python3 /scripts/04_score_masked.py --base_dir /projects
🔬 QA Control (Step 4)
Run the sharpness and filter reports to validate your automated culling.

Check Filter Performance:

Bash

python3 /scripts/04c_qa_filter_report.py --base_dir /projects
What to look for: Open 04c_filter_qa_report.html. This is the "Wall of Shame." Review the images that FAILED the test. Did the MAX_CONTOUR_COUNT filter correctly catch images with extra objects? Did the MIN_SHARPNESS_THRESHOLD correctly catch blurry images? If good images are failing, your filters are too strict.

Check Sharpness Spectrum:

Bash

python3 /scripts/04d_qa_sharpness_spectrum.py --base_dir /projects
What to look for: Open 04d_sharpness_spectrum_report.html. This is your "blurryness" test. Look at the "Worst 10 (Blurriest)" section. Are these images still acceptably sharp? Or are they blurry and low-quality? If you find the "Worst" images are still too blurry, you should increase the MIN_SHARPNESS_THRESHOLD value at the top of 04_score_masked.py to be more strict.

Step 5: Replace Backgrounds
This pastes your sharp, filtered subjects from Step 4 onto your 00_background_library images, applying a light blur.

Bash

python3 /scripts/05_replace_backgrounds.py --base_dir /projects
🔬 QA Control (Step 5)
Run the final manual cull report.

Bash

python3 /scripts/05b_qa_backgrounds.py --base_dir /projects
What to look for: Open 05b_background_qa_report.html. This is your final human-in-the-loop cull. Look at the composited images. Do they look "natural"? Is the BLUR_RADIUS (set in 05_replace_backgrounds.py) creating a good "pop"?

If you see any image that "feels wrong" (bad composite, weird pose, etc.), copy its mv command and paste it into your terminal to disqualify it.

Step 6: Caption Final Images
This uses Vertex AI to create captions for the images you just approved. The prompt is engineered to ignore backgrounds and tattoos/watermarks.

Bash

python3 /scripts/05_caption.py \
  --base_dir /projects \
  --project_id "your-gcp-project-id" \
  --trigger_words "your trigger, "
🔬 QA Control (Step 6)
Run the final caption review report.

Bash

python3 /scripts/06b_qa_captions.py --base_dir /projects
What to look for: Open 06b_caption_qa_report.html. Read the captions.

Are they neutral? Did the AI successfully ignore the background? (You should not see "mountains," "pigeons," etc.)

Are they clean? Did the AI successfully ignore tattoos/watermarks? (You should not see the word "tattoo".)

If any captions are bad, use the mv command to cull the image/text pair.

Once this step is complete, your dataset in /projects/05_training_data is fully validated and ready for training.
