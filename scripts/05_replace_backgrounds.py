#!/usr/bin/env python3

import os
from pathlib import Path
from shutil import rmtree, copy2
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter
import random
import logging
import argparse

# --- Configuration ---
IMAGE_QUALITY = 95

# --- NEW: Background Blur ---
# Controls how blurry the background will be.
# 2.0 is a light blur, 5.0 is a stronger blur.
# Let's start with 3.0
BLUR_RADIUS = 3.0
# --- End Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_random_background(background_files):
    """Selects one random background file path."""
    return random.choice(background_files)

def prepare_background(bg_path, target_w, target_h):
    """
    Loads a background, randomly resizes/crops it, and applies a blur.
    """
    try:
        with Image.open(bg_path).convert("RGB") as bg_img:
            bg_w, bg_h = bg_img.size
            
            # 1. Ensure background is larger than target
            if bg_w < target_w or bg_h < target_h:
                bg_img = ImageOps.fit(bg_img, (target_w, target_h), Image.LANCZOS)
                bg_w, bg_h = bg_img.size

            # 2. Randomly crop a region
            max_x = bg_w - target_w
            max_y = bg_h - target_h
            
            rand_x = random.randint(0, max_x)
            rand_y = random.randint(0, max_y)
            
            crop_box = (rand_x, rand_y, rand_x + target_w, rand_y + target_h)
            background_crop = bg_img.crop(crop_box)
            
            # --- NEW: Apply Gaussian Blur ---
            blurred_background = background_crop.filter(
                ImageFilter.GaussianBlur(radius=BLUR_RADIUS)
            )
            # --- END NEW ---
            
            return blurred_background
            
    except Exception as e:
        logging.warning(f"Could not load or crop background {bg_path.name}: {e}")
        # Return a failsafe black (and blurred) background
        black_bg = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        return black_bg.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

def main():
    parser = argparse.ArgumentParser(description="Replace backgrounds from a library with blur.")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        required=True, 
        help="The base project directory (e.g., /projects/tara_tainton_TEST)"
    )
    args = parser.parse_args()

    BASE_DIR = Path(args.base_dir)
    SOURCE_DIR = BASE_DIR / "04_top100_staging"
    OUTPUT_DIR = BASE_DIR / "05_training_data"
    BACKGROUND_LIB_DIR = BASE_DIR.parent / "00_background_library"
    
    if not BACKGROUND_LIB_DIR.is_dir():
        BACKGROUND_LIB_DIR = BASE_DIR / "00_background_library"

    logging.info("--- Starting Step 5: Random Background Replacement (with Blur) ---")
    
    if not SOURCE_DIR.is_dir():
        logging.error(f"*** FATAL ERROR: Input dir not found: {SOURCE_DIR} ***")
        return
        
    if not BACKGROUND_LIB_DIR.is_dir():
        logging.error(f"*** FATAL ERROR: Background library not found: {BACKGROUND_LIB_DIR} ***")
        return

    logging.info(f"Scanning for backgrounds in: {BACKGROUND_LIB_DIR}")
    background_files = list(BACKGROUND_LIB_DIR.glob("*.jpg")) + \
                       list(BACKGROUND_LIB_DIR.glob("*.jpeg")) + \
                       list(BACKGROUND_LIB_DIR.glob("*.png"))
                       
    if not background_files:
        logging.error(f"*** FATAL ERROR: No .jpg or .png backgrounds found in library. ***")
        return
        
    logging.info(f"Found {len(background_files)} backgrounds. Applying blur radius: {BLUR_RADIUS}")
    
    source_folders = [f for f in SOURCE_DIR.iterdir() if f.is_dir()]
    
    if not source_folders:
        logging.warning(f"No category folders found in {SOURCE_DIR}. Nothing to do.")
        return

    for source_folder in source_folders:
        output_folder = OUTPUT_DIR / source_folder.name
        
        logging.info(f"\nProcessing {source_folder.name} -> {output_folder.name}")
        
        if output_folder.exists():
            logging.info(f"Cleaning old directory: {output_folder}")
            rmtree(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        image_files = list(source_folder.glob("*.png"))

        if not image_files:
            logging.warning("No .png images found in this folder. Skipping.")
            continue

        for img_path in tqdm(image_files, desc=f"Replacing backgrounds in {source_folder.name}"):
            try:
                with Image.open(img_path).convert("RGBA") as foreground:
                    img_w, img_h = foreground.size

                    random_bg_path = get_random_background(background_files)
                    
                    # --- This now returns a BLURRED background ---
                    blurred_background_crop = prepare_background(random_bg_path, img_w, img_h)
                    
                    # 3. Composite the image
                    # Paste the SHARP foreground onto the BLURRED background
                    blurred_background_crop.paste(foreground, (0, 0), foreground)
                    
                    # 4. Save as a high-quality JPEG
                    output_filename = output_folder / f"{img_path.stem}.jpg"
                    blurred_background_crop.save(output_filename, "JPEG", quality=IMAGE_QUALITY)

                # 5. Copy the caption file, if one exists
                txt_path = img_path.with_suffix(".txt")
                if txt_path.exists():
                    copy2(txt_path, output_folder / txt_path.name)

            except Exception as e:
                tqdm.write(f"Error processing {img_path.name}: {e}")

    logging.info("\n--- Step 5 Complete ---")
    logging.info(f"Your final (blurred bg) training images are in: {OUTPUT_DIR}")
    logging.info("These folders are now ready for captioning.")

if __name__ == "__main__":
    main()
