#!/usr/bin/env python3

import subprocess
import sys
import json
import math
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import argparse
import shutil
import logging

# --- Configuration ---
VIDEO_SOURCE_DIR = Path("/projects/source_videos_high_res")
OUTPUT_DIR = Path("/projects/01_source_frames")

# We can use multiple CPU workers. This is safe and fast.
MAX_WORKERS = 10 

FRAMES_PER_SECOND = 1
JPEG_QUALITY = 2
# --- End Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_video_duration(video_path):
    """Uses ffprobe to get the duration of a video."""
    command = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(video_path)
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                return float(stream.get("duration", 0.0))
    except Exception:
        return 0.0
    return 0.0

def process_video(video_file):
    """
    Uses ffmpeg (CPU-ONLY) to extract NATIVE resolution frames.
    This is the robust, "tried and tested" method.
    """
    duration = get_video_duration(video_file)
    expected_frames = int(math.ceil(duration * FRAMES_PER_SECOND))
    tqdm.write(f"  [STARTING] {video_file.name} ({duration:.2f}s, expecting {expected_frames} frames)...")

    video_name = video_file.stem
    video_output_folder = OUTPUT_DIR / video_name
    video_output_folder.mkdir(parents=True, exist_ok=True)

    output_pattern = video_output_folder / f"{video_name}_frame_%06d.jpg"

    # --- THIS IS THE ROBUST CPU-ONLY COMMAND ---
    # All '-hwaccel' flags have been removed.
    command = [
        "ffmpeg",
        "-i", str(video_file),
        "-r", str(FRAMES_PER_SECOND),
        "-q:v", str(JPEG_QUALITY),
        "-hide_banner",
        "-loglevel", "error",
        str(output_pattern)
    ]
    # --- END OF FIX ---

    try:
        subprocess.run(command, check=True)
        return f"  [DONE] Successfully processed {video_name}"
    except Exception as e:
        return f"  [ERROR] CPU processing failed for {video_file.name}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Extract native-res frames.")
    parser.add_argument(
        "--clean_output", action="store_true",
        help="Wipe the OUTPUT_DIR clean before starting."
    )
    args = parser.parse_args()

    if not VIDEO_SOURCE_DIR.is_dir():
        print(f"Error: Video source directory not found at {VIDEO_SOURCE_DIR}")
        return

    if args.clean_output and OUTPUT_DIR.exists():
        print(f"Cleaning output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for videos in: {VIDEO_SOURCE_DIR}")
    video_files = list(VIDEO_SOURCE_DIR.rglob("*.mp4")) + \
                  list(VIDEO_SOURCE_DIR.rglob("*.mov")) + \
                  list(VIDEO_SOURCE_DIR.rglob("*.m4v")) + \
                  list(VIDEO_SOURCE_DIR.rglob("*.avi")) + \
                  list(VIDEO_SOURCE_DIR.rglob("*.mkv")) + \
                  list(VIDEO_SOURCE_DIR.rglob("*.webm"))

    if not video_files:
        print("No video files found. Exiting.")
        return

    print(f"Found {len(video_files)} videos. Starting CPU-based frame extraction...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_video = {executor.submit(process_video, vf): vf for vf in video_files}

        for future in tqdm(concurrent.futures.as_completed(future_to_video), total=len(video_files), desc="Extracting frames"):
            video = future_to_video[future]
            try:
                result = future.result()
                tqdm.write(result) # Print the [DONE] or [ERROR] message
            except Exception as exc:
                print(f'{video.name} generated an exception: {exc}')

    print("\n--- Native Frame Extraction Complete ---")
    print(f"All native frames are located in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
