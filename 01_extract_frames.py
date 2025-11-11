#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import argparse  # <-- Import argparse
import shutil    # <-- Import shutil

# --- Configuration ---
# MODIFIED: Target the clean, high-res triage folder
VIDEO_SOURCE_DIR = Path("/projects/source_videos_high_res") 
OUTPUT_DIR = Path("/projects/source_images")
MAX_WORKERS = 10
FRAMES_PER_SECOND = 1
JPEG_QUALITY = 2
# --- End Configuration ---

def process_video(video_file):
    """
    Uses ffmpeg to extract NATIVE resolution frames from a single video file.
    """
    video_name = video_file.stem
    video_output_folder = OUTPUT_DIR / video_name
    video_output_folder.mkdir(parents=True, exist_ok=True)
    
    # Naming convention is critical for downstream scripts
    output_pattern = video_output_folder / f"{video_name}_frame_%06d.jpg"

    # --- MODIFIED: The "-vf" scale/crop line has been REMOVED ---
    command = [
        "ffmpeg",
        "-i", str(video_file),
        "-r", str(FRAMES_PER_SECOND),
        "-q:v", str(JPEG_QUALITY),
        "-hide_banner",
        "-loglevel", "error",
        str(output_pattern)
    ]

    try:
        subprocess.run(command, check=True)
        return f"Successfully processed {video_name}"
    except subprocess.CalledProcessError as e:
        return f"ERROR processing {video_name}: {e}"
    except Exception as e:
        return f"UNEXPECTED ERROR processing {video_name}: {e}"

def main():
    # --- ADDED: Argparse for --clean_output ---
    parser = argparse.ArgumentParser(description="Extract native-res frames.")
    parser.add_argument(
        "--clean_output",
        action="store_true",
        help="Wipe the OUTPUT_DIR clean before starting."
    )
    args = parser.parse_args()
    # --- END ADDED ---

    if not VIDEO_SOURCE_DIR.is_dir():
        print(f"Error: Video source directory not found at {VIDEO_SOURCE_DIR}")
        return

    # --- ADDED: Clean output logic ---
    if args.clean_output and OUTPUT_DIR.exists():
        print(f"Cleaning output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    # --- END ADDED ---

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

    print(f"Found {len(video_files)} videos. Starting native frame extraction...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_video = {executor.submit(process_video, vf): vf for vf in video_files}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_video), total=len(video_files), desc="Extracting frames"):
            video = future_to_video[future]
            try:
                result = future.result()
                if "ERROR" in result:
                    print(result)
            except Exception as exc:
                print(f'{video.name} generated an exception: {exc}')
                
    print("\n--- Native Frame Extraction Complete ---")
    print(f"All native frames are located in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
