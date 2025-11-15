#!/usr/bin/env python3

import os
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_html_report(image_data, output_path, base_project_dir, disqualified_dir):
    """Generates the HTML report for final background-replaced images."""
    
    html = f"""
    <html><head><title>Actionable Culling Report (Backgrounds)</title>
    <style>
        body {{ font-family: sans-serif; background: #222; color: #eee; margin: 20px; }}
        h1 {{ text-align: center; border-bottom: 2px solid #555; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #444; padding: 10px; background: #333; }}
        .container {{ width: 98%; margin: auto; }}
        .category {{ margin-bottom: 50px; }}
        .item-grid {{ display: flex; flex-wrap: wrap; justify-content: center; }}
        .item {{
            display: inline-block; width: 300px; margin: 10px;
            background: #333; border-radius: 8px; overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3); vertical-align: top;
            border: 1px solid #444;
        }}
        .item img {{ 
            width: 100%; height: 300px; object-fit: cover; 
        }}
        .item p {{ 
            margin: 10px; font-size: 0.8em; word-wrap: break-word; 
            text-align: left;
        }}
        .item b {{ color: #8af; }}
        /* --- NEW: Style for the 'mv' command --- */
        .item .cull-command {{
            background: #111;
            color: #f99; /* Changed to a 'warning' color */
            padding: 8px;
            font-family: monospace;
            word-wrap: break-word;
            border-radius: 4px;
            margin-top: 10px;
            display: block;
            user-select: all; /* Makes it easy to copy */
        }}
    </style>
    </head><body>
    <div class="container">
    <h1>Actionable Culling Report (Backgrounds)</h1>
    <p style="text-align: center;">
        Review these final images. If any are bad, copy its 
        <code>mv</code> command and paste it into your container terminal to disqualify it.
    </p>
    """
    
    for category, items in image_data.items():
        html += f"<div class='category'><h2>Category: {category} ({len(items)} images)</h2>"
        html += "<div class='item-grid'>"
        
        for item_path in items:
            try:
                # Get relative path for the <img> tag
                rel_path = item_path.relative_to(base_project_dir)
            except ValueError:
                rel_path = item_path
            
            # --- NEW: Get absolute container paths for the 'mv' command ---
            container_src_path = Path("/projects") / rel_path
            
            # Define the destination path
            container_dest_dir = Path("/projects") / disqualified_dir.relative_to(base_project_dir) / category
            container_dest_path = container_dest_dir / item_path.name
            
            # The command to move the file
            mv_command = f"mv {container_src_path} {container_dest_path}"
                
            html += f"""
            <div class='item'>
                <img src='{rel_path}' alt='{item_path.name}' loading='lazy'>
                <p>
                    <b>File:</b> {item_path.name}<br>
                    <b>Disqualify (Copy this):</b>
                    <span class='cull-command'>{mv_command}</span>
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
    parser = argparse.ArgumentParser(description="Generate a QA gallery for background replacement.")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        required=True, 
        help="The base project directory (e.g., /projects/tara_tainton_TEST)"
    )
    args = parser.parse_args()
    
    BASE_DIR = Path(args.base_dir)
    SOURCE_DIR = BASE_DIR / "05_training_data"
    REPORT_FILE = BASE_DIR / "05b_background_qa_report.html"
    
    # --- NEW: Define the disqualified directory ---
    DISQUALIFIED_DIR = BASE_DIR / "05_disqualified_backgrounds"
    DISQUALIFIED_DIR.mkdir(exist_ok=True)
    
    if not SOURCE_DIR.is_dir():
        logging.error(f"FATAL: Source dir not found: {SOURCE_DIR}")
        logging.error("Please run 05_replace_backgrounds.py first.")
        return

    logging.info("--- Starting Step 5b: Actionable Culling Report Generator ---")
    
    image_data = {}
    source_folders = [f for f in SOURCE_DIR.iterdir() if f.is_dir()]

    if not source_folders:
        logging.warning(f"No category folders found in {SOURCE_DIR}. Nothing to do.")
        return

    for category_dir in source_folders:
        category_name = category_dir.name
        logging.info(f"Scanning category: {category_name}")
        
        # --- NEW: Create matching disqualified sub-folder ---
        (DISQUALIFIED_DIR / category_name).mkdir(exist_ok=True)
        
        images = sorted(list(category_dir.glob("*.jpg")))
        image_data[category_name] = images

    logging.info(f"All categories processed. Generating report at {REPORT_FILE}...")
    # --- NEW: Pass the disqualified dir to the report generator ---
    generate_html_report(image_data, REPORT_FILE, BASE_DIR, DISQUALIFIED_DIR)
    
    logging.info("--- Actionable Culling Report Complete ---")
    logging.info(f"Disqualified files will be moved to: {DISQUALIFIED_DIR.name}")
    logging.info(f"To view the report, exit the container and open this file on your host:")
    logging.info(f"file://{REPORT_FILE.absolute()}")

if __name__ == "__main__":
    main()

