#!/usr/bin/env python3

import os
from pathlib import Path
import json
import logging
import argparse
import math

# --- Configuration ---
SAMPLE_SIZE = 10 # Show this many images from each group
# --- End Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_html_report(spectrum_data, output_path, base_project_dir):
    """Generates the HTML report for the sharpness spectrum."""
    
    html = f"""
    <html><head><title>Sharpness Spectrum Report (Best/Middle/Worst)</title>
    <style>
        body {{ font-family: sans-serif; background: #222; color: #eee; margin: 20px; }}
        h1 {{ text-align: center; border-bottom: 2px solid #555; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #444; padding: 10px; background: #333; }}
        h3 {{ border-bottom: 1px solid #444; padding: 8px; margin-top: 30px; }}
        h3.pass {{ background: #363; color: #9f9; }}
        h3.middle {{ background: #553; color: #ff9; }}
        h3.fail {{ background: #633; color: #f99; }}
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
            background-image: 
                linear-gradient(45deg, #444 25%, transparent 25%), 
                linear-gradient(-45deg, #444 25%, transparent 25%), 
                linear-gradient(45deg, transparent 75%, #444 75%), 
                linear-gradient(-45deg, transparent 75%, #444 75%);
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
        }}
        .item p {{ margin: 10px; font-size: 0.8em; word-wrap: break-word; }}
        .item b {{ color: #8af; }}
        .item .sharp-score {{ color: #8f8; font-weight: bold; font-size: 1.1em; }}
        .item .blur-score {{ color: #f88; font-weight: bold; font-size: 1.1em; }}
    </style>
    </head><body>
    <div class="container">
    <h1>Sharpness Spectrum Report (Best/Middle/Worst)</h1>
    <p style="text-align: center;">This report shows the sharpness spread for all images that PASSED the mask filter.</p>
    """
    
    for category, data in spectrum_data.items():
        html += f"<div class='category'><h2>Category: {category} (Found {data['count']} passed images)</h2>"
        
        # --- 1. Best (Sharpest) ---
        html += "<h3 class='pass'>Best {SAMPLE_SIZE} (Sharpest)</h3><div class='item-grid'>"
        for item in data['best']:
            s = item['scores']
            details = (
                f"<b class='sharp-score'>Raw Sharpness: {s['raw_sharpness']:.0f}</b><br><hr>"
                f"<b>File:</b> {item['file']}<br>"
                f"<b>S_Pose:</b> {s['raw_pose']:.3f}<br>"
                f"<b>S_Bright:</b> {s['raw_brightness']:.3f}"
            )
            try:
                rel_path = Path(item['path']).relative_to(base_project_dir)
            except ValueError:
                rel_path = item['path']
            html += f"<div class='item'><img src='{rel_path}' alt='{item['file']}' loading='lazy'><p>{details}</p></div>"
        html += "</div>"

        # --- 2. Middle (Average) ---
        html += f"<h3 class='middle'>Middle {SAMPLE_SIZE} (Average)</h3><div class='item-grid'>"
        for item in data['middle']:
            s = item['scores']
            details = (
                f"<b>Raw Sharpness: {s['raw_sharpness']:.0f}</b><br><hr>"
                f"<b>File:</b> {item['file']}<br>"
                f"<b>S_Pose:</b> {s['raw_pose']:.3f}<br>"
                f"<b>S_Bright:</b> {s['raw_brightness']:.3f}"
            )
            try:
                rel_path = Path(item['path']).relative_to(base_project_dir)
            except ValueError:
                rel_path = item['path']
            html += f"<div class='item'><img src='{rel_path}' alt='{item['file']}' loading='lazy'><p>{details}</p></div>"
        html += "</div>"

        # --- 3. Worst (Blurriest) ---
        html += f"<h3 class='fail'>Worst {SAMPLE_SIZE} (Blurriest)</h3><div class='item-grid'>"
        for item in data['worst']:
            s = item['scores']
            details = (
                f"<b class='blur-score'>Raw Sharpness: {s['raw_sharpness']:.0f}</b><br><hr>"
                f"<b>File:</b> {item['file']}<br>"
                f"<b>S_Pose:</b> {s['raw_pose']:.3f}<br>"
                f"<b>S_Bright:</b> {s['raw_brightness']:.3f}"
            )
            try:
                rel_path = Path(item['path']).relative_to(base_project_dir)
            except ValueError:
                rel_path = item['path']
            html += f"<div class='item'><img src='{rel_path}' alt='{item['file']}' loading='lazy'><p>{details}</p></div>"
        html += "</div>"
        
        html += "</div>" # End category

    html += "</div></body></html>"
    
    try:
        output_path.write_text(html)
        logging.info(f"Successfully generated report at {output_path}")
    except Exception as e:
        logging.error(f"Error writing HTML report: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate a sharpness spectrum QA report.")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        required=True, 
        help="The base project directory (e.g., /projects/tara_tainton_TEST)"
    )
    args = parser.parse_args()
    
    BASE_DIR = Path(args.base_dir)
    JSON_FILE = BASE_DIR / "scoring_results_yolo_masked.json"
    REPORT_FILE = BASE_DIR / "04d_sharpness_spectrum_report.html"

    if not JSON_FILE.exists():
        logging.error(f"FATAL: JSON file not found: {JSON_FILE}")
        logging.error("Please run 04_score_masked.py first.")
        return

    logging.info("--- Starting Step 4d: Sharpness Spectrum Report Generator ---")
    
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)
        
    spectrum_data = {}
    
    for category, items in data.items():
        logging.info(f"Processing category: {category}")
        
        # 1. Get *only* the images that passed the mask filter
        passed_items = [item for item in items if item['final_score'] > 0.0]
        
        if not passed_items:
            logging.warning(f"  No images passed the mask filter for {category}. Skipping.")
            spectrum_data[category] = {"best": [], "middle": [], "worst": [], "count": 0}
            continue
            
        # 2. Sort them by *sharpness* (highest to lowest)
        sorted_by_sharpness = sorted(
            passed_items, 
            key=lambda x: x['scores']['raw_sharpness'], 
            reverse=True
        )
        
        n = len(sorted_by_sharpness)
        logging.info(f"  Found {n} passed images. Sampling for spectrum...")
        
        # 3. Get the three sample groups
        best_samples = sorted_by_sharpness[:SAMPLE_SIZE]
        
        middle_index = max(0, (n // 2) - (SAMPLE_SIZE // 2))
        middle_samples = sorted_by_sharpness[middle_index : middle_index + SAMPLE_SIZE]
        
        worst_samples = sorted_by_sharpness[-SAMPLE_SIZE:]
        
        spectrum_data[category] = {
            "best": best_samples,
            "middle": middle_samples,
            "worst": worst_samples,
            "count": n
        }

    logging.info(f"All categories processed. Generating report at {REPORT_FILE}...")
    generate_html_report(spectrum_data, REPORT_FILE, BASE_DIR)
    
    logging.info("--- Sharpness Spectrum Report Complete ---")
    logging.info(f"To view the report, exit the container and open this file on your host:")
    logging.info(f"file://{REPORT_FILE.absolute()}")

if __name__ == "__main__":
    main()
