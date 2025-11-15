#!/usr/bin/env python3

import os
import subprocess
import json
from pathlib import Path
from tqdm import tqdm
import logging

# --- Configuration ---
VIDEO_SOURCE_DIR = Path("/projects/source_videos_high_res")
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
FRAMES_PER_SECOND = 1.0 # From your 01_extract.py
# --- End Configuration ---

logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_video_info(video_path):
    """
    Uses ffprobe to get width, height, and duration.
    Returns (width, height, duration_in_seconds)
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

        # Find the first video stream
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            return None, None, 0.0

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        duration = float(video_stream.get("duration", 0.0))

        return width, height, duration

    except Exception as e:
        logging.warning(f"  WARN: Could not probe file {video_path.name}. Error: {e}")
        return None, None, 0.0

def main():
    logging.info("--- Starting Video Library Audit ---")
    if not VIDEO_SOURCE_DIR.is_dir():
        logging.error(f"FATAL: Source directory not found: {VIDEO_SOURCE_DIR}")
        return

    video_files = list(VIDEO_SOURCE_DIR.rglob("*.mp4")) + \
                  list(VIDEO_SOURCE_DIR.rglob("*.mov")) + \
                  list(VIDEO_SOURCE_DIR.rglob("*.m4v")) + \
                  list(VIDEO_SOURCE_DIR.rglob("*.avi")) + \
                  list(VIDEO_SOURCE_DIR.rglob("*.mkv")) + \
                  list(VIDEO_SOURCE_DIR.rglob("*.webm"))

    if not video_files:
        logging.error(f"No video files found in {VIDEO_SOURCE_DIR}.")
        return

    logging.info(f"Found {len(video_files)} total video files. Analyzing...")
    logging.info("=" * 40)

    total_hd_videos = 0
    total_expected_frames = 0
    total_other_videos = 0

    for video_path in tqdm(video_files, desc="Auditing Videos"):
        width, height, duration = get_video_info(video_path)

        if width == TARGET_WIDTH and height == TARGET_HEIGHT:
            total_hd_videos += 1
            # Round up to nearest whole number for frame count
            frames = int(math.ceil(duration * FRAMES_PER_SECOND))
            total_expected_frames += frames
            tqdm.write(f"  [HD] {video_path.name} - {duration:.2f}s = {frames} frames")
        elif width is not None:
            total_other_videos += 1
            tqdm.write(f"  [Other] {video_path.name} - {width}x{height}")

    logging.info("=" * 40)
    logging.info("--- Audit Complete ---")
    logging.info(f"Total HD (1920x1080) Videos: {total_hd_videos}")
    logging.info(f"Total Other Videos: {total_other_videos}")
    logging.info(f"TOTAL EXPECTED FRAMES (at 1fps): {total_expected_frames}")
    logging.info("=" * 40)

if __name__ == "__main__":
    # Need to import math for this script
    import math
    main()
