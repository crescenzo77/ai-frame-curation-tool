#!/usr/bin/env python3

import os
from pathlib import Path
import logging
import argparse
import random
import cv2
from tqdm import tqdm

# --- Configuration ---
RANDOM_SAMPLE_SIZE = 100 # Show this many random images
TOP_N_SHARPEST = 100     # Show this many sharpest images
# --- End Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_sharpness_score(image_path):
    """Calculates a global sharpness score for an image."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return 0.0
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(image_gray, cv2.CV_64F).var()
    except Exception:
        return 0.0

def generate_html_report(random_sample, sharpest_sample, output_path, base_project_dir):
    """Generates the HTML report for rejected files."""
    
    html = f"""
    <html><head><title>Reject Bin QA Report</title>
    <style>
        body {{ font-family: sans-serif; background: #222; color: #eee; margin: 20px; }}
        h1 {{ text-align: center; border-bottom: 2px solid #555; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #444; padding: 10px; background: #333; }}
        .container {{ width: 98%; margin: auto; }}
        .category {{ margin-bottom: 50px; }}
        .item-grid {{ display: flex; flex-wrap: wrap; justify-content: center; }}
        .item {{
            display: inline-block; width: 250px; margin: 10px;
            background: #333; border-radius: 8px; overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3); vertical-align: top;
            border: 1px solid #444;
        }}
        .item img {{ 
            width: 100%; height: 250px; object-fit: cover; 
            border-bottom: 1px solid #444;
        }}
        .item p {{ margin: 10px; font-size: 0.8em; word-wrap: break-word; text-align: left; }}
        .item b {{ color: #8af; }}
        .item .sharp-score {{ color: #8f8; font-weight: bold; }}
    </style>
    </head><body>
    <div class="container">
    <h1>Reject Bin QA Report ('no_person_detected')</h1>
    
    <div class='category'>
        <h2>Control 1: Random Sample (Sample of {len(random_sample)})</h2>
        <div class='item-grid'>
    """
    
    for img_path in random_sample:
        try:
            rel_path = img_path.relative_to(base_project_dir)
        except ValueError:
            rel_path = img_path
        html += f"""
        <div class='item'>
            <img src='{rel_path}' alt='{img_path.name}' loading='lazy'>
            <p><b>File:</b> {img_path.name}</p>
        </div>
        """
    html += "</div></div>" # End item-grid and category

    # --- Section 2: Top N Sharpest ---
    html += f"""
    <div class='category'>
        <h2>Control 2: Top {len(sharpest_sample)} Sharpest (False Negative Finder)</h2>
        <div class='item-grid'>
    """
    
    for score, img_path in sharpest_sample:
        try:
            rel_path = img_path.relative_to(base_project_dir)
        except ValueError:
            rel_path = img_path
        html += f"""
        <div class='item'>
            <img src='{rel_path}' alt='{img_path.name}' loading='lazy'>
            <p>
                <b>File:</b> {img_path.name}<br>
                <span class='sharp-score'>Sharpness: {score:.0f}</span>
            </p>
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
    parser = argparse.ArgumentParser(description="Generate a QA report for rejected frames.")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        required=True, 
        help="The base project directory (e.g., /projects/tara_tainton_TEST)"
    )
    args = parser.parse_args()
    
    BASE_DIR = Path(args.base_dir)
    # We will focus on the most likely folder for false negatives
    REJECT_DIR = BASE_DIR / "02_sorted_candidates" / "no_person_detected"
    REPORT_FILE = BASE_DIR / "02b_reject_qa_report.html"

    if not REJECT_DIR.is_dir():
        logging.error(f"FATAL: Reject dir not found: {REJECT_DIR}")
        logging.error("Please run 02_sort.py first.")
        return

    logging.info("--- Starting Step 2b: Reject Bin QA Report Generator ---")
    
    all_rejects = list(REJECT_DIR.glob("*.jpg")) + \
                    list(REJECT_DIR.glob("*.jpeg")) + \
                    list(REJECT_DIR.glob("*.png"))

    if not all_rejects:
        logging.warning(f"No rejected images found in {REJECT_DIR}. Nothing to do.")
        return

    logging.info(f"Found {len(all_rejects)} rejected images.")
    
    # --- 1. Control 1: Random Sample ---
    logging.info(f"Taking a random sample of {RANDOM_SAMPLE_SIZE} images...")
    if len(all_rejects) <= RANDOM_SAMPLE_SIZE:
        random_sample = all_rejects
    else:
        random_sample = random.sample(all_rejects, RANDOM_SAMPLE_SIZE)
    
    # --- 2. Control 2: Top N Sharpest ---
    logging.info("Scoring all rejected images for sharpness (this may take a moment)...")
    all_scores = []
    for img_path in tqdm(all_rejects, desc="Scoring rejects"):
        score = get_sharpness_score(img_path)
        all_scores.append((score, img_path))
        
    # Sort by score, highest to lowest
    all_scores.sort(key=lambda x: x[0], reverse=True)
    
    sharpest_sample = all_scores[:TOP_N_SHARPEST]

    logging.info(f"All processing complete. Generating report at {REPORT_FILE}...")
    generate_html_report(random_sample, sharpest_sample, REPORT_FILE, BASE_DIR)
    
    logging.info("--- Reject QA Report Complete ---")
    logging.info(f"To view the report, exit the container and open this file on your host:")
    logging.info(f"file://{REPORT_FILE.absolute()}")

if __name__ == "__main__":
    main()
