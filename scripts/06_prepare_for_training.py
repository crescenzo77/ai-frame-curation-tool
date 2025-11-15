#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import logging

# --- Configuration ---
# This is the output from your 04_ and 05_ scripts
SOURCE_DIR = Path("/projects/04_final_dataset")

# This is the final input for the Kohya trainer
KOHYA_DIR = Path("/projects/05_training_data")

# Define the mapping:
# "clean_source_folder": "kohya_output_folder_with_repeats"
CATEGORY_MAP = {
    "face_and_hair": "15_tara_tainton_face",
    "upper_body": "10_tara_tainton_upper",
    "full_body": "5_tara_tainton_full"
}
# --- End Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info(f"Starting Kohya preparation...")

    if KOHYA_DIR.exists():
        logging.warning(f"Cleaning existing Kohya prep dir: {KOHYA_DIR}")
        shutil.rmtree(KOHYA_DIR)

    KOHYA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created empty dir: {KOHYA_DIR}")

    total_files_copied = 0

    for source_name, kohya_name in CATEGORY_MAP.items():
        source_folder = SOURCE_DIR / source_name
        kohya_output_path = KOHYA_DIR / kohya_name

        if not source_folder.is_dir():
            logging.warning(f"Source folder not found: {source_folder}. Skipping.")
            continue

        kohya_output_path.mkdir()
        logging.info(f"Processing '{source_folder.name}' -> '{kohya_name}'")

        # Find all images and their matching .txt files
        image_files = list(source_folder.glob("*.jpg")) + \
                      list(source_folder.glob("*.jpeg")) + \
                      list(source_folder.glob("*.png"))

        if not image_files:
            logging.warning(f"No images found in {source_folder}. Skipping.")
            continue

        file_count = 0
        for img_path in tqdm(image_files, desc=f"Copying {kohya_name}"):
            txt_path = img_path.with_suffix(".txt")

            if not txt_path.exists():
                logging.warning(f"  Missing caption for {img_path.name}! Skipping file.")
                continue

            # Copy both files
            shutil.copy(img_path, kohya_output_path / img_path.name)
            shutil.copy(txt_path, kohya_output_path / txt_path.name)
            file_count += 1

        logging.info(f"Copied {file_count} image/caption pairs for {kohya_name}.")
        total_files_copied += file_count

    if total_files_copied == 0:
        logging.error("No files were copied. The trainer will fail.")
    else:
        logging.info(f"\n--- Kohya Preparation Complete ---")
        logging.info(f"Total files prepared: {total_files_copied}")
        logging.info(f"Data is ready in: {KOHYA_DIR}")

if __name__ == "__main__":
    main()
