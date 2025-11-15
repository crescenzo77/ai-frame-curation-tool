#!/usr/bin/env python3

import os
from pathlib import Path
import json
import logging
import argparse

# --- Configuration ---
TOP_N_PASSED = 50 # Show this many "good" images
# --- End Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_html_report(failed_items, passed_items, output_path, base_project_dir):
    """Generates the HTML report for failed and passed items."""
    
    html = f"""
    <html><head><title>Mask QA Filter Report (A/B Test)</title>
    <style>
        body {{ font-family: sans-serif; background: #222; color: #eee; margin: 20px; }}
        h1 {{ text-align: center; border-bottom: 2px solid #555; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #444; padding: 10px; background: #333; }}
        h2.fail {{ background: #633; color: #f99; }}
        h2.pass {{ background: #363; color: #9f9; }}
        .container {{ width: 98%; margin: auto; }}
        .category {{ margin-bottom: 50px; }}
        .item-grid {{ display: flex; flex-wrap: wrap; justify-content: center; }}
        .item {{
            display: inline-block; width: 250px; margin: 10px;
            background: #333; border-radius: 8px; overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3); vertical-align: top;
            border: 1px solid #444;
        }}
        .item.fail {{ border: 2px solid #f77; }}
        .item.pass {{ border: 2px solid #7f7; }}
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
        .item .fail-reason {{ color: #f88; font-weight: bold; font-size: 1.1em; }}
    </style>
    </head><body>
    <div class="container">
    <h1>Mask QA Filter Report ("Wall of Shame" vs. "Wall of Stars")</h1>
    """
    
    # --- 1. The "Wall of Shame" (Failed Items) ---
    for category, items in failed_items.items():
        if not items: continue
        
        # --- THIS IS THE FIX (Line 59) ---
        html += f"<div class='category'><h2 class='fail'>Category: {category} — FAILED Images ({len(items)} found)</h2><div class='item-grid'>"
        # --- END FIX ---
        
        for item in items:
            s = item['scores']
            details = (
                f"<b>File:</b> {item['file']}<br>"
                f"<b class='fail-reason'>Reason: {s.get('mask_status', 'FAIL')}</b><br>"
                f"<b>Source:</b> {item['source_video']}<br>"
                f"<b>Frame:</b> {item['frame_index']}"
            )
            try:
                rel_path = Path(item['path']).relative_to(base_project_dir)
            except ValueError:
                rel_path = item['path']

            html += f"""
            <div class='item fail'>
                <img src='{rel_path}' alt='{item['file']}' loading='lazy'>
                <p>{details}</p>
            </div>
            """
        html += "</div></div>" # End item-grid and category

    # --- 2. The "Wall of Stars" (Passed Items) ---
    for category, items in passed_items.items():
        if not items: continue

        # --- THIS IS THE FIX (Line 91) ---
        html += f"<div class='category'><h2 class='pass'>Category: {category} — Top {len(items)} PASSED Images</h2><div class='item-grid'>"
        # --- END FIX ---
        
        for i, item in enumerate(items):
            s = item['scores']
            details = (
                f"<b>Rank:</b> {i+1}<br>"
                f"<b>Final Score:</b> {item['final_score']:.4f}<br><hr>"
                f"<b>File:</b> {item['file']}<br>"
                f"<b>Mask QA:</b> {s.get('mask_status', 'PASS')}<br>"
                f"<b>S_Pose:</b> {s['raw_pose']:.3f}<br>"
                f"<b>S_Sharp:</b> {s.get('S_sharpness', 0):.3f}"
            )
            try:
                rel_path = Path(item['path']).relative_to(base_project_dir)
            except ValueError:
                rel_path = item['path']

            html += f"""
            <div class='item pass'>
                <img src='{rel_path}' alt='{item['file']}' loading='lazy'>
                <p>{details}</p>
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
    parser = argparse.ArgumentParser(description="Generate a QA report for the hard filter.")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        required=True, 
        help="The base project directory (e.g., /projects/tara_tainton_TEST)"
    )
    args = parser.parse_args()
    
    BASE_DIR = Path(args.base_dir)
    JSON_FILE = BASE_DIR / "scoring_results_yolo_masked.json"
    REPORT_FILE = BASE_DIR / "04c_filter_qa_report.html"

    if not JSON_FILE.exists():
        logging.error(f"FATAL: JSON file not found: {JSON_FILE}")
        logging.error("Please run 04_score_masked.py first.")
        return

    logging.info("--- Starting Step 4c: Filter QA Report Generator ---")
    
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)
        
    failed_items = {}
    passed_items = {}
    
    for category, items in data.items():
        logging.info(f"Processing category: {category}")
        
        # We use 'final_score' to find the failed items
        cat_failed = [item for item in items if item['final_score'] == 0.0]
        
        # The 'passed' items are everything else, already sorted by score
        cat_passed = [item for item in items if item['final_score'] > 0.0]
        
        failed_items[category] = cat_failed
        passed_items[category] = cat_passed[:TOP_N_PASSED] # Show top N passed
        
        logging.info(f"  Found {len(cat_failed)} FAILED images.")
        logging.info(f"  Found {len(cat_passed)} PASSED images (showing top {TOP_N_PASSED}).")

    logging.info(f"All categories processed. Generating report at {REPORT_FILE}...")
    generate_html_report(failed_items, passed_items, REPORT_FILE, BASE_DIR)
    
    logging.info("--- Filter QA Report Complete ---")
    logging.info(f"To view the report, exit the container and open this file on your host:")
    logging.info(f"file://{REPORT_FILE.absolute()}")

if __name__ == "__main__":
    main()
