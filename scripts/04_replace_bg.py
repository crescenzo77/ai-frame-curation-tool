#!/usr/bin/env python3

import os
from pathlib import Path
from shutil import rmtree, copy2
from tqdm import tqdm
from rembg import remove # This uses the default 'u2net' model
from PIL import Image
import random
import io
import logging

# --- Configuration ---
BASE_PROJECT_DIR = Path("/projects")
COLLAGE_FILE = BASE_PROJECT_DIR / "collage.jpg"

# 1. (FIXED) Source folders are the new Top 100 staging dir
SOURCE_DIR = BASE_PROJECT_DIR / "03_top100_staging"
SOURCE_FOLDERS = [
    SOURCE_DIR / "face_and_hair",
    SOURCE_DIR / "upper_body",
    SOURCE_DIR / "full_body"
]

# 2. (FIXED) The clean output folder
OUTPUT_DIR = BASE_PROJECT_DIR / "04_final_dataset"
# --- End Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("--- Starting Phase 4: FAST Background Replacement (Collage) ---")

    if not COLLAGE_FILE.exists():
        logging.error(f"*** FATAL ERROR: Collage file not found at {COLLAGE_FILE} ***")
        return

    try:
        Image.MAX_IMAGE_PIXELS = None 
        collage_pil = Image.open(COLLAGE_FILE).convert("RGB")
        collage_w, collage_h = collage_pil.size
        logging.info(f"Collage loaded successfully ({collage_w}x{collage_h}).")
    except Exception as e:
        logging.error(f"*** FATAL ERROR: Could not load collage image. Error: {e} ***")
        return

    for source_folder in SOURCE_FOLDERS:
        if not source_folder.is_dir():
            logging.warning(f"Source folder {source_folder} not found. Skipping.")
            continue

        # (FIXED) Map the source folder to the output folder
        if "face_and_hair" in source_folder.name:
            output_folder = OUTPUT_DIR / "face_and_hair"
        elif "upper_body" in source_folder.name:
            output_folder = OUTPUT_DIR / "upper_body"
        elif "full_body" in source_folder.name:
            output_folder = OUTPUT_DIR / "full_body"
        else:
            logging.warning(f"Skipping unknown folder {source_folder.name}")
            continue

        logging.info(f"\nProcessing {source_folder.name} -> {output_folder.name}")

        if output_folder.exists():
            logging.info(f"Cleaning old directory: {output_folder}")
            rmtree(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        image_files = list(source_folder.glob("*.jpg")) + \
                      list(source_folder.glob("*.jpeg")) + \
                      list(source_folder.glob("*.png")) + \
                      list(source_folder.glob("*.webp"))

        if not image_files:
            logging.warning("No images found in this folder. Skipping.")
            continue

        for img_path in tqdm(image_files, desc=f"Replacing backgrounds in {source_folder.name}"):
            try:
                with open(img_path, 'rb') as f_in:
                    input_bytes = f_in.read()

                output_bytes = remove(input_bytes)

                foreground = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
                img_w, img_h = foreground.size

                if img_w > collage_w or img_h > collage_h:
                    background_crop = collage_pil.resize((img_w, img_h), Image.LANCZOS)
                else:
                    rand_x = random.randint(0, collage_w - img_w)
                    rand_y = random.randint(0, collage_h - img_h)
                    background_crop = collage_pil.crop((rand_x, rand_y, rand_x + img_w, rand_y + img_h))

                new_image = Image.new("RGB", (img_w, img_h))
                new_image.paste(background_crop, (0, 0))
                new_image.paste(foreground, (0, 0), foreground)

                output_filename = output_folder / f"{img_path.name}"
                new_image.save(output_filename, "JPEG", quality=95)

                txt_path = img_path.with_suffix(".txt")
                if txt_path.exists():
                    copy2(txt_path, output_folder / txt_path.name)

            except Exception as e:
                if "_idat" in str(e):
                    tqdm.write(f"SKIPPED (Pillow error): {img_path.name}. This image may be corrupt.")
                else:
                    tqdm.write(f"Error processing {img_path.name}: {e}")

    logging.info("\n--- Phase 4 Complete ---")
    logging.info("FAST background replacement finished.")
    logging.info(f"Your final (pre-cull) images are in: {OUTPUT_DIR}")
    logging.info("These folders are now ready for your manual review and cull.")

if __name__ == "__main__":
    main()
