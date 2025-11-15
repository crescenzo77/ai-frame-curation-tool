#!/usr/bin/env python3

import os
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_html_report(image_data, output_path, base_project_dir, disqualified_dir):
    """Generates the HTML report for final caption review."""
    
    html = f"""
    <html><head><title>Actionable Caption QA Report</title>
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
            border-bottom: 1px solid #444; /* Added separator for image */
        }}
        .item .caption {{
            background: #2a2a2a;
            color: #ddd;
            padding: 10px;
            font-family: monospace;
            font-size: 0.9em;
            min-height: 80px; /* Ensure consistent height for captions */
            max-height: 150px;
            overflow-y: auto;
            border-bottom: 1px solid #444; /* Separator for caption */
        }}
        .item .cull-command {{
            background: #111;
            color: #f99;
            padding: 8px;
            margin: 10px; /* Adjust margin to keep it tight */
            font-family: monospace;
            word-wrap: break-word;
            border-radius: 4px;
            display: block;
            user-select: all; /* Makes it easy to copy */
            font-size: 0.8em; /* Slightly smaller font for command */
        }}
    </style>
    </head><body>
    <div class="container">
    <h1>Actionable Caption QA Report</h1>
    <p style="text-align: center;">
        Review these images and their generated captions. If any are bad, 
        copy its <code>mv</code> command to disqualify the pair.
        <br><strong>Remember to run copied commands inside your docker container.</strong>
    </p>
    """
    
    for category, items in image_data.items():
        html += f"<div class='category'><h2>Category: {category} ({len(items)} images)</h2>"
        html += "<div class='item-grid'>"
        
        for img_path, caption in items:
            try:
                # Get relative path for the <img> tag for web display
                rel_path = img_path.relative_to(base_project_dir)
            except ValueError:
                rel_path = img_path
            
            # Get absolute container paths for the 'mv' command
            # The base_project_dir is mounted at /projects inside the container
            container_src_path_img = Path("/projects") / rel_path
            container_src_path_txt = container_src_path_img.with_suffix(".txt")
            
            # Define the destination path for disqualified items
            # This path is relative to the BASE_DIR, so we need to rebuild it
            container_dest_base_dir_rel = Path("06_disqualified_captions")
            container_dest_dir = Path("/projects") / container_dest_base_dir_rel / category
            
            # Ensure the destination directory exists in the container if command is run
            # The script should create it, but this adds a safety net.
            # mv_command_mkdir = f"mkdir -p {container_dest_dir}; "
            
            container_dest_path_img = container_dest_dir / img_path.name
            container_dest_path_txt = container_dest_dir / container_src_path_txt.name
            
            # The command to move BOTH the image and the text file
            # Add an mkdir -p to ensure the destination directory exists if not already created
            mv_command = (
                f"mkdir -p {container_dest_dir} && "  # Ensures dest category dir exists
                f"mv '{container_src_path_img}' '{container_dest_path_img}' && "
                f"mv '{container_src_path_txt}' '{container_dest_path_txt}'"
            )
                
            html += f"""
            <div class='item'>
                <img src='{rel_path}' alt='{img_path.name}' loading='lazy'>
                <div class='caption'>{caption}</div>
                <span class='cull-command'>{mv_command}</span>
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
    parser = argparse.ArgumentParser(description="Generate a QA gallery for captions.")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        required=True, 
        help="The base project directory (e.g., /projects/tara_tainton_TEST)"
    )
    args = parser.parse_args()
    
    BASE_DIR = Path(args.base_dir)
    SOURCE_DIR = BASE_DIR / "05_training_data"
    REPORT_FILE = BASE_DIR / "06b_caption_qa_report.html"
    
    # Define the disqualified directory
    DISQUALIFIED_DIR = BASE_DIR / "06_disqualified_captions"
    DISQUALIFIED_DIR.mkdir(exist_ok=True) # Ensure base disqualified dir exists
    
    if not SOURCE_DIR.is_dir():
        logging.error(f"FATAL: Source dir not found: {SOURCE_DIR}")
        logging.error("Please run 05_caption.py first.")
        return

    logging.info("--- Starting Step 6b: Actionable Caption Report Generator ---")
    
    image_data = {}
    source_folders = [f for f in SOURCE_DIR.iterdir() if f.is_dir()]

    if not source_folders:
        logging.warning(f"No category folders found in {SOURCE_DIR}. Nothing to do.")
        return

    for category_dir in source_folders:
        category_name = category_dir.name
        logging.info(f"Scanning category: {category_name}")
        
        # Create matching disqualified sub-folder if it doesn't exist
        (DISQUALIFIED_DIR / category_name).mkdir(exist_ok=True)
        
        image_files = sorted(list(category_dir.glob("*.jpg"))) + \
                      sorted(list(category_dir.glob("*.jpeg"))) + \
                      sorted(list(category_dir.glob("*.png"))) + \
                      sorted(list(category_dir.glob("*.webp")))

        category_items = []
        
        for img_path in image_files:
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                caption = txt_path.read_text()
            else:
                caption = "--- ERROR: CAPTION FILE MISSING ---"
            
            category_items.append((img_path, caption))
            
        image_data[category_name] = category_items

    logging.info(f"All categories processed. Generating report at {REPORT_FILE}...")
    generate_html_report(image_data, REPORT_FILE, BASE_DIR, DISQUALIFIED_DIR)
    
    logging.info("--- Actionable Caption Report Complete ---")
    logging.info(f"Disqualified pairs will be moved to: {DISQUALIFIED_DIR.name}")
    logging.info(f"To view the report, exit the container and open this file on your host:")
    logging.info(f"file://{REPORT_FILE.absolute()}")

if __name__ == "__main__":
    main()
