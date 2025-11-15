#!/usr/bin/env python3

import os
from pathlib import Path
from shutil import rmtree
from tqdm import tqdm
from rembg import remove # Uses 'u2net' model
from PIL import Image
import io
import logging
import argparse

# --- Configuration ---
# All paths are now dynamic
# --- End Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Mask subjects using rembg.")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        required=True, 
        help="The base project directory (e.g., /projects/tara_tainton_TEST)"
    )
    args = parser.parse_args()
    
    BASE_DIR = Path(args.base_dir)
    SOURCE_DIR = BASE_DIR / "02_sorted_candidates"
    OUTPUT_DIR = BASE_DIR / "03_masked_subjects"

    # We only care about the folders we want to score
    SOURCE_FOLDERS_TO_MASK = [
        SOURCE_DIR / "face_and_hair",
        SOURCE_DIR / "upper_body",
        SOURCE_DIR / "full_body"
    ]

    logging.info("--- Starting Phase 3: Subject Masking (rembg) ---")
    
    if OUTPUT_DIR.exists():
        logging.warning(f"Cleaning existing masked directory: {OUTPUT_DIR}")
        rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for source_folder in SOURCE_FOLDERS_TO_MASK:
        if not source_folder.is_dir():
            logging.warning(f"Source folder {source_folder} not found. Skipping.")
            continue
            
        # Create matching output folder
        output_folder = OUTPUT_DIR / source_folder.name
        output_folder.mkdir(parents=True, exist_ok=True)

        logging.info(f"Processing {source_folder.name} -> {output_folder.name}")

        image_files = list(source_folder.glob("*.jpg")) + \
                      list(source_folder.glob("*.jpeg")) + \
                      list(source_folder.glob("*.png")) + \
                      list(source_folder.glob("*.webp"))

        if not image_files:
            logging.warning("No images found in this folder. Skipping.")
            continue

        for img_path in tqdm(image_files, desc=f"Masking subjects in {source_folder.name}"):
            try:
                with open(img_path, 'rb') as f_in:
                    input_bytes = f_in.read()

                # Run rembg to get a PNG with transparent background
                output_bytes = remove(input_bytes)

                # Verify the image is valid before saving
                with Image.open(io.BytesIO(output_bytes)) as img:
                    img.verify()

                # Save the new PNG file
                # We change the extension to .png to preserve transparency
                output_filename = output_folder / f"{img_path.stem}.png"
                with open(output_filename, 'wb') as f_out:
                    f_out.write(output_bytes)

            except Exception as e:
                tqdm.write(f"Error processing {img_path.name}: {e}")

    logging.info("\n--- Phase 3 Masking Complete ---")
    logging.info(f"Masked PNGs are ready for scoring in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
