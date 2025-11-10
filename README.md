# README

The v2 sorter uses LDM, L2Net, LoFTR, automatic U-Body, upper-body, sorts, scores, and de-duplicates images to find the best available frames, solving common dataset problems like poor face focus and false faces.

### Key features:

* **Face/Keypointter Sorting:** Gate to disqualify images with tiny, distant poses from the upper_body and full_body training pools, ensuring higher quality results for the scoring.
* **Advanced Focus Scoring:** (Implements a v-sort model that can distinguish between real faces and images that might look like faces (e.g., drawings, statues). It also ensures the sharpest object in the frame, penalizing images with blurry faces/sharp backgrounds.)
* **"False-Face" Filter:** [Note: The description for this appears merged with "Advanced Focus Scoring" in the original text. I've grouped them as they appear.]
* **Duplicate/Near-Duplicate Guardrail:** Utilizes a high-accuracy de-duplicator, combining SI-ren frame hashing and a Phash filter to remove similar images, resulting in a cleaner, more diverse dataset.
* **Video-Temporal Quote:** A frame hard-cap for body categories that forces selections to be from different times in the video (e.g., 1 image from 1 video, 3 from another, etc.), preventing the system from locking on to a single scene.
* **"Pose-Weighted" Scoring:** The "magic" step. It uses a custom-trained EfficientNetv2 (based on the work of AI-APES 'Consistent-Factor' & 'Recursive'), The Pose 'How-To' dataset, MediaPipe, PyTorch/Nix combined with Docker, making it 100% reproducible.

### How it works:

* **1. Process:** The Pipeline `01_extract.py` runs a face detect (U-Body), keypoint LDM, and a LoFTR v2 model on all images, saving the results in `datastore/datastore.json`. Output is 100k+ lines (e.g., 240,000 images). `02_sort.py` creates the `datastore.json` file. The 240,000 images are organized into subfolders (e.g., `datastore/videos_folder_1/output_images_0`).
* **2. Process:** The `03_score_and_rank_v2.py` script applies the 30% pose area gate. If a full_body or upper_body candidate's pose core is less than 30% of the image, it's disqualified. It then runs the `v4` "False-Face" Filter, keeping the scoring pools clean. [The text continues with more detail on the scoring process].

### Quick Start:

* Clone the repo: `git clone https://github.com/crescendoai/frame-curation-tool`
* `pip install -r requirements.txt`
* Add your secrets (Google service account key file) to the root of the project. Never commit this file.
* Place your .mp4, .mov, etc. files into the `datastore/videos_folder_1`.
* Run the Docker image (or start the Python script) and start the Gradio/WebUI by navigating to your Docker container IP (or `127.0.0.1:7860`).
* Go to `http://127.0.0.1:7860` in your browser.
* Extract frames (This will take a while):
    ```bash
    python3 01_extract.py
    ```

---

### Step 2: Sort all frames (This will also take a while)

This v2 sorter uses the 30% pose area gate.

```bash
python3 02_sort.py
