#!/usr/bin/env python3

import os
import subprocess
import json
import math
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
from PIL import Image

# --- Configuration ---
FRAMES_PER_SECOND = 1.0 # Matches your Mac extraction script
# --- End Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_video_info(video_path):
    """
    Uses ffprobe to get duration.
    Returns (duration_in_seconds)
    """
    command = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(video_path)
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                return float(stream.get("duration", 0.0))
    except Exception as e:
        logging.warning(f"  WARN: Could not probe file {video_path.name}. Error: {e}")
        return 0.0
    return 0.0

def validate_frames(frame_folder):
    """
    Counts files and checks for corruption using a lightweight method.
    Returns (actual_frame_count, corrupt_frame_count)
    """
    if not frame_folder.is_dir():
        return 0, 0
    
    # We'll check for .jpg as that's what your Mac script outputs
    frame_files = list(frame_folder.glob("*.jpg"))
    actual_count = len(frame_files)
    corrupt_count = 0
    
    if actual_count == 0:
        return 0, 0
        
    for frame_path in frame_files:
        try:
            # 1. Check for zero-byte files (very fast)
            if frame_path.stat().st_size == 0:
                corrupt_count += 1
                continue
                
            # 2. Check for truncated/corrupt JPEG (fast, non-intensive)
            # This is what you requested: a non-intensive check
            with Image.open(frame_path) as img:
                img.verify() # Reads header and metadata, but not pixel data
                
        except Exception as e:
            # This catches zero-byte files and corrupt/truncated images
            corrupt_count += 1
            
    return actual_count, corrupt_count

def generate_html_report(report_data, output_path):
    """Generates the HTML report you requested."""
    
    html = f"""
    <html><head><title>Frame Extraction Validation Report</title>
    <style>
        body {{ font-family: sans-serif; background: #222; color: #eee; margin: 20px; }}
        h1 {{ text-align: center; border-bottom: 2px solid #555; padding-bottom: 10px; }}
        table {{ width: 90%; margin: 20px auto; border-collapse: collapse; }}
        th, td {{ border: 1px solid #444; padding: 10px 12px; text-align: left; }}
        th {{ background: #333; }}
        .status-ok {{ color: #7f7; }}
        .status-warn {{ color: #ff7; }}
        .status-error {{ color: #f77; font-weight: bold; }}
    </style>
    </head><body>
    <h1>Frame Extraction Validation Report</h1>
    <table>
    <tr>
        <th>Video File</th>
        <th>Duration (s)</th>
        <th>Expected Frames (at 1fps)</th>
        <th>Actual Frames Extracted</th>
        <th>Corrupt/Empty Frames</th>
        <th>Status (Summary)</th>
    </tr>
    """
    
    total_expected = 0
    total_actual = 0
    total_corrupt = 0
    
    # Sort data by status to bring errors to the top
    report_data.sort(key=lambda x: x['status'])
    
    for item in report_data:
        total_expected += item['expected']
        total_actual += item['actual']
        total_corrupt += item['corrupt']
        
        status_text = item['status']
        status_class = 'status-ok'
        if "Warning" in status_text:
            status_class = 'status-warn'
        if "Error" in status_text:
            status_class = 'status-error'

        html += f"""
        <tr>
            <td>{item['video_name']}</td>
            <td>{item['duration']:.2f}s</td>
            <td>{item['expected']}</td>
            <td>{item['actual']}</td>
            <td>{item['corrupt']}</td>
            <td class="{status_class}">{status_text}</td>
        </tr>
        """
    
    # Add a summary footer
    status_class = 'status-ok'
    if total_expected != total_actual: status_class = 'status-warn'
    if total_corrupt > 0: status_class = 'status-error'
    
    html += f"""
    <tr style="background: #333; font-weight: bold;">
        <td>TOTALS ({len(report_data)} videos)</td>
        <td>-</td>
        <td>{total_expected}</td>
        <td>{total_actual}</td>
        <td>{total_corrupt}</td>
        <td class="{status_class}">Totals Check</td>
    </tr>
    """
    
    html += "</table></body></html>"
    
    try:
        output_path.write_text(html)
    except Exception as e:
        logging.error(f"Error writing HTML report: {e}")

def main():
    parser = argparse.ArgumentParser(description="Validate extracted frames against source videos.")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        required=True, 
        help="The base project directory (e.g., /projects/tara_tainton)"
    )
    args = parser.parse_args()
    
    BASE_DIR = Path(args.base_dir)
    # This path comes from your Mac script
    VIDEO_SOURCE_DIR = BASE_DIR / "source_videos_high_res"
    # This path also comes from your Mac script
    FRAME_OUTPUT_DIR = BASE_DIR / "01_source_frames"
    REPORT_FILE = BASE_DIR / "01b_validation_report.html"
    
    if not VIDEO_SOURCE_DIR.is_dir():
        logging.error(f"FATAL: Source video directory not found: {VIDEO_SOURCE_DIR}")
        return
    if not FRAME_OUTPUT_DIR.is_dir():
        logging.error(f"FATAL: Frame output directory not found: {FRAME_OUTPUT_DIR}")
        logging.error("This script requires the '01_source_frames' directory to exist.")
        return

    video_files = sorted(
        list(VIDEO_SOURCE_DIR.rglob("*.mp4")) + \
        list(VIDEO_SOURCE_DIR.rglob("*.mov")) + \
        list(VIDEO_SOURCE_DIR.rglob("*.m4v")) + \
        list(VIDEO_SOURCE_DIR.rglob("*.avi")) + \
        list(VIDEO_SOURCE_DIR.rglob("*.mkv")) + \
        list(VIDEO_SOURCE_DIR.rglob("*.webm"))
    )
    
    if not video_files:
        logging.error(f"No video files found in {VIDEO_SOURCE_DIR}")
        return
        
    logging.info(f"Found {len(video_files)} videos. Auditing against {FRAME_OUTPUT_DIR}...")
    
    report_data = []

    for video_path in tqdm(video_files, desc="Validating Extractions"):
        video_name = video_path.stem
        duration = get_video_info(video_path)
        
        if duration == 0:
            report_data.append({
                "video_name": video_path.name,
                "duration": 0, "expected": 0, "actual": 0, "corrupt": 0,
                "status": "Error: Could not read video"
            })
            continue

        # Use math.ceil, just like the original 01_extract.py script
        expected_frames = int(math.ceil(duration * FRAMES_PER_SECOND))
        
        frame_folder = FRAME_OUTPUT_DIR / video_name
        actual_frames, corrupt_frames = validate_frames(frame_folder)
        
        # Determine status
        status = "OK"
        if actual_frames == 0 and expected_frames > 0:
            status = "Error: Missing extraction folder"
        elif corrupt_frames > 0:
            status = f"Error: Corrupt ({corrupt_frames} files)"
        elif actual_frames != expected_frames:
            diff = actual_frames - expected_frames
            # Note: Your ffmpeg script might have a 1-frame-off issue
            status = f"Warning: Mismatch (Expected {expected_frames}, Got {actual_frames}, Diff: {diff})"
            
        report_data.append({
            "video_name": video_path.name,
            "duration": duration,
            "expected": expected_frames,
            "actual": actual_frames,
            "corrupt": corrupt_frames,
            "status": status
        })

    logging.info(f"Validation complete. Generating report at {REPORT_FILE}...")
    generate_html_report(report_data, REPORT_FILE)
    
    logging.info("--- Validation Complete ---")
    logging.info(f"To view the report, exit the container and open this file on your host:")
    logging.info(f"file://{REPORT_FILE}")

if __name__ == "__main__":
    main()
