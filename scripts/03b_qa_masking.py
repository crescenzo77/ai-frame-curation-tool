#!/usr/bin/env python3

import os
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
import random

# --- Configuration ---
SAMPLE_SIZE = 100 # Show this many random images per category
# --- End Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_html_report(comparison_data, output_path, base_project_dir):
    """Generates the HTML report for side-by-side comparison."""
    
    html = f"""
    <html><head><title>Masking QA Report (Side-by-Side)</title>
    <style>
        body {{ font-family: sans-serif; background: #222; color: #eee; margin: 20px; }}
        h1 {{ text-align: center; border-bottom: 2px solid #555; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #444; padding: 10px; background: #333; }}
        .container {{ width: 98%; margin: auto; }}
        .category {{ margin-bottom: 50px; }}
        .item-grid {{ display: flex; flex-wrap: wrap; justify-content: center; }}
        .item {{
            display: inline-block; width: 600px; margin: 15px;
            background: #2a2a2a; border-radius: 8px; overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3); vertical-align: top;
            border: 1px solid #444;
        }}
        .item .image-pair {{ display: flex; }}
        .item img {{ 
            width: 50%; height: 300px; object-fit: cover; 
        }}
        /* Checkerboard background for the masked (PNG) image */
        .item img.masked {{
            background-image: 
                linear-gradient(45deg, #444 25%, transparent 25%), 
                linear-gradient(-45deg, #444 25%, transparent 25%), 
                linear-gradient(45deg, transparent 75%, #444 75%), 
                linear-gradient(-45deg, transparent 75%, #444 75%);
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
        }}
        .item p {{ 
            margin: 10px; font-size: 0.9em; word-wrap: break-word; 
            text-align: center; background: #333; padding: 8px;
        }}
        .item b {{ color: #8af; }}
    </style>
    </head><body>
    <div class="container">
    <h1>Masking QA Report (Side-by-Side)</h1>
    """
    
    for category, items in comparison_data.items():
        html += f"<div class='category'><h2>Category: {category} (Sample of {len(items)})</h2>"
        html += "<div class='item-grid'>"
        
        for item in items:
            html += f"""
            <div class='item'>
                <div class='image-pair'>
                    <img src='{item['original_rel_path']}' alt='Original'>
                    <img src='{item['masked_rel_path']}' alt='Masked' class='masked'>
                </div>
                <p><b>File:</b> {item['filename']}</p>
            </div>
            """
        
        html += "</div></div>" # End item-grid and category
    
    html += "</div></body></html>"
    
    try:
        output_path.write_text(html)
        logging.info(f"Successfully generated report at {output_path}")
    except Exception as e:
        logging.error(f"Error writing HTML report: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate a side-by-side QA report for rembg masking.")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        required=True, 
        help="The base project directory (e.g., /projects/tara_tainton_TEST)"
    )
    args = parser.parse_args()
    
    BASE_DIR = Path(args.base_dir)
    ORIGINAL_DIR = BASE_DIR / "02_sorted_candidates"
    MASKED_DIR = BASE_DIR / "03_masked_subjects"
    REPORT_FILE = BASE_DIR / "03b_mask_qa_report.html"

    if not ORIGINAL_DIR.is_dir():
        logging.error(f"FATAL: Original dir not found: {ORIGINAL_DIR}")
        return
    if not MASKED_DIR.is_dir():
        logging.error(f"FATAL: Masked dir not found: {MASKED_DIR}")
        logging.error("Please run 03_mask_subjects.py first.")
        return

    logging.info("--- Starting Step 3b: Masking QA Report Generator ---")
    
    categories_to_check = [
        "face_and_hair",
        "upper_body",
        "full_body"
    ]
    
    comparison_data = {}

    for category in categories_to_check:
        logging.info(f"Processing category: {category}")
        original_cat_dir = ORIGINAL_DIR / category
        masked_cat_dir = MASKED_DIR / category
        
        if not original_cat_dir.is_dir() or not masked_cat_dir.is_dir():
            logging.warning(f"Skipping category {category}: one or both directories missing.")
            continue
            
        # Find all the *original* files (JPEGs)
        original_files = list(original_cat_dir.glob("*.jpg")) + \
                         list(original_cat_dir.glob("*.jpeg"))
        
        if not original_files:
            logging.warning(f"No original JPG/JPEG files found in {original_cat_dir}")
            continue
            
        # Take a random sample if the list is too large
        if len(original_files) > SAMPLE_SIZE:
            logging.info(f"Sampling {SAMPLE_SIZE} images from {len(original_files)}...")
            image_sample = random.sample(original_files, SAMPLE_SIZE)
        else:
            image_sample = original_files
            
        category_items = []
        
        for orig_path in tqdm(image_sample, desc=f"Comparing {category}"):
            # The masked file has the same stem but a .png extension
            masked_path = masked_cat_dir / f"{orig_path.stem}.png"
            
            if masked_path.exists():
                try:
                    # Get paths relative to the base for the HTML report
                    original_rel_path = orig_path.relative_to(BASE_DIR)
                    masked_rel_path = masked_path.relative_to(BASE_DIR)
                    
                    category_items.append({
                        "filename": orig_path.name,
                        "original_rel_path": original_rel_path,
                        "masked_rel_path": masked_rel_path
                    })
                except ValueError:
                    logging.warning(f"Path error for {orig_path.name}, skipping.")
            else:
                logging.warning(f"Missing masked file for {orig_path.name}, skipping.")
        
        comparison_data[category] = category_items

    logging.info(f"All categories processed. Generating report at {REPORT_FILE}...")
    generate_html_report(comparison_data, REPORT_FILE, BASE_DIR)
    
    logging.info("--- Masking QA Report Complete ---")
    logging.info(f"To view the report, exit the container and open this file on your host:")
    logging.info(f"file://{REPORT_FILE.absolute()}")

if __name__ == "__main__":
    main()
