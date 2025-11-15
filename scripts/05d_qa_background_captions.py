#!/usr/bin/env python3

import os
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_html_report(image_data, output_path, base_project_dir):
    """Generates the HTML report for background caption review."""
    
    html = f"""
    <html><head><title>Background Library Caption QA</title>
    <style>
        body {{ font-family: sans-serif; background: #222; color: #eee; margin: 20px; }}
        h1 {{ text-align: center; border-bottom: 2px solid #555; padding-bottom: 10px; }}
        .container {{ width: 98%; margin: auto; }}
        .item-grid {{ display: flex; flex-wrap: wrap; justify-content: center; }}
        .item {{
            display: inline-block; width: 300px; margin: 10px;
            background: #333; border-radius: 8px; overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3); vertical-align: top;
            border: 1px solid #444;
        }}
        .item img {{ 
            width: 100%; height: 300px; object-fit: cover; 
            border-bottom: 1px solid #444;
        }}
        .item .caption {{
            background: #2a2a2a;
            color: #ddd;
            padding: 10px;
            font-family: monospace;
            font-size: 0.9em;
            height: 80px;
            overflow-y: auto;
        }}
        .item .filename {{
            background: #111;
            color: #8af;
            padding: 8px;
            margin: 10px;
            font-family: monospace;
            word-wrap: break-word;
            border-radius: 4px;
            display: block;
            user-select: all;
            font-size: 0.8em;
        }}
    </style>
    </head><body>
    <div class="container">
    <h1>Background Library Caption QA Report</h1>
    <p style="text-align: center;">
        Review these captions. Identify images where the caption is "noisy" (describes specific objects).<br>
        These "noisy" images will be added to a list for stronger blurring.
    </p>
    <div class='item-grid'>
    """
    
    for img_path, caption in image_data:
        try:
            # Get relative path for the <img> tag
            rel_path = img_path.relative_to(base_project_dir)
        except ValueError:
            rel_path = img_path
            
        html += f"""
        <div class='item'>
            <img src='{rel_path}' alt='{img_path.name}' loading='lazy'>
            <div class='caption'>{caption}</div>
            <span class='filename'>{img_path.name}</span>
        </div>
        """
    
    html += "</div></div></body></html>" # End item-grid, container, and body
    
    try:
        output_path.write_text(html)
        logging.info(f"Successfully generated report at {output_path}")
    except Exception as e:
        logging.error(f"Error writing HTML report: {e}")

def main():
    # This script doesn't need args, it has one job
    BASE_DIR = Path("/projects") # We are running inside the container
    SOURCE_DIR = BASE_DIR / "00_background_library"
    REPORT_FILE = BASE_DIR / "05d_background_caption_report.html"

    if not SOURCE_DIR.is_dir():
        logging.error(f"FATAL: Source dir not found: {SOURCE_DIR}")
        return

    logging.info("--- Starting Step 5d: Background Caption Report Generator ---")
    
    image_files = sorted(list(SOURCE_DIR.glob("*.jpg"))) + \
                  sorted(list(SOURCE_DIR.glob("*.jpeg"))) + \
                  sorted(list(SOURCE_DIR.glob("*.png"))) + \
                  sorted(list(SOURCE_DIR.glob("*.webp")))

    image_data = []
    
    if not image_files:
        logging.warning(f"No image files found in {SOURCE_DIR}. Nothing to do.")
        return

    for img_path in image_files:
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            caption = txt_path.read_text()
        else:
            caption = "--- ERROR: CAPTION FILE MISSING ---"
        
        image_data.append((img_path, caption))

    logging.info(f"All {len(image_data)} backgrounds processed. Generating report...")
    generate_html_report(image_data, REPORT_FILE, BASE_DIR)
    
    logging.info("--- Background Caption Report Complete ---")
    logging.info(f"To view the report, exit the container and open this file on your host:")
    logging.info(f"file:///mnt/storage/comfyui_storage/projects/tara_tainton/05d_background_caption_report.html")

if __name__ == "__main__":
    main()
