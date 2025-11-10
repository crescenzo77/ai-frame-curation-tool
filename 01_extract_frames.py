import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# --- Configuration ---

# 1. Set this to the directory containing your ORIGINAL video files.
#    This script will scan it recursively.
VIDEO_SOURCE_DIR = Path("/projects/source_videos")

# 2. This is where the new 1024x1024 frames will be saved.
OUTPUT_DIR = Path("/projects/source_images")

# 3. How many videos to process at once. Adjust based on your CPU (Ryzen 9 7900X)
#    A good starting point is (total_cores - 2). 10 is likely safe.
MAX_WORKERS = 10

# 4. Frame extraction rate (e.g., 1 = 1 frame per second)
FRAMES_PER_SECOND = 1

# 5. Output image quality (1 = best, 31 = worst)
JPEG_QUALITY = 2

# --- End Configuration ---

def process_video(video_file):
    """
    Uses ffmpeg to extract 1024x1024 center-cropped frames from a single video file.
    """
    
    # Create a unique subfolder for this video's frames
    video_name = video_file.stem
    video_output_folder = OUTPUT_DIR / video_name
    video_output_folder.mkdir(parents=True, exist_ok=True)
    
    output_pattern = video_output_folder / f"{video_name}_frame_%06d.jpg"

    # --- The FFMPEG Command Explained ---
    # -i {video_file}: Input file
    # -vf "..."      : Video Filter
    #   scale=w=1024:h=1024:force_original_aspect_ratio=increase
    #     - Scales the video. Tells ffmpeg to make the SHORTEST side 1024px
    #       (e.g., 1280x720 -> 1820x1024)
    #   crop=w=1024:h=1024:x=(in_w-1024)/2:y=(in_h-1024)/2
    #     - Performs a 1024x1024 center crop.
    # -r {FRAMES_PER_SECOND}: Extract this many frames per second
    # -q:v {JPEG_QUALITY} : Set the JPEG quality
    # -hide_banner -loglevel error: Keeps the console output clean
    
    command = [
        "ffmpeg",
        "-i", str(video_file),
        "-vf", "scale=w=1024:h=1024:force_original_aspect_ratio=increase,crop=w=1024:h=1024:x=(in_w-1024)/2:y=(in_h-1024)/2",
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
    if not VIDEO_SOURCE_DIR.is_dir():
        print(f"Error: Video source directory not found at {VIDEO_SOURCE_DIR}")
        print("Please update the VIDEO_SOURCE_DIR variable in the script.")
        return

    # Create the main output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for videos in: {VIDEO_SOURCE_DIR}")
# --- REPLACE IT WITH THIS ---
    video_files = list(VIDEO_SOURCE_DIR.rglob("*.mp4")) + \
                  list(VIDEO_SOURCE_DIR.rglob("*.mov")) + \
                  list(VIDEO_SOURCE_DIR.rglob("*.m4v")) + \
                  list(VIDEO_SOURCE_DIR.rglob("*.avi"))
                  
    if not video_files:
        print("No video files found. Exiting.")
        return

    print(f"Found {len(video_files)} videos. Starting extraction...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Use tqdm to create a progress bar
        future_to_video = {executor.submit(process_video, vf): vf for vf in video_files}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_video), total=len(video_files), desc="Extracting frames"):
            video = future_to_video[future]
            try:
                result = future.result()
                if "ERROR" in result:
                    print(result) # Print errors to console
            except Exception as exc:
                print(f'{video.name} generated an exception: {exc}')
                
    print("\n--- Frame Extraction Complete ---")
    print(f"All 1024x1024 frames are located in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
