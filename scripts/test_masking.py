#!/usr/bin/env python3

from pathlib import Path
from rembg import remove, new_session
from PIL import Image
import logging

# --- Configuration ---

# 1. The image we want to test
INPUT_FILE = Path("/projects/c_full_body_top80_yolo/tara_video_006_frame_000748.jpg")

# 2. The two models we are testing
MODEL_DEFAULT = "u2net"
MODEL_ADVANCED = "isnet-general-use"  # <-- THIS IS THE FIX (one dash)

# 3. The output files for our side-by-side comparison
OUTPUT_DEFAULT = Path("/projects/TEST_RESULT_DEFAULT_u2net.png")
OUTPUT_ADVANCED = Path("/projects/TEST_RESULT_ADVANCED_isnet.png")

# --- End Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run_test(model_name, output_path, session):
    try:
        logging.info(f"Testing model: {model_name}...")

        with open(INPUT_FILE, 'rb') as f_in:
            input_bytes = f_in.read()

        # Run the segmentation
        output_bytes = remove(input_bytes, session=session)

        # Save the result
        with open(output_path, 'wb') as f_out:
            f_out.write(output_bytes)

        logging.info(f"SUCCESS! Saved to: {output_path}")

    except Exception as e:
        logging.error(f"Failed to test model {model_name}: {e}")

def main():
    if not INPUT_FILE.exists():
        logging.error(f"FATAL: Cannot find input file: {INPUT_FILE}")
        logging.error("Please find a problem image and update the INPUT_FILE path in this script.")
        return

    # Test 1: The default, fast model
    run_test(MODEL_DEFAULT, OUTPUT_DEFAULT, new_session(MODEL_DEFAULT))

    # Test 2: The advanced, high-quality model
    logging.info(f"Downloading/loading model: {MODEL_ADVANCED}...")
    adv_session = new_session(MODEL_ADVANCED)
    logging.info("Model loaded.")
    run_test(MODEL_ADVANCED, OUTPUT_ADVANCED, adv_session)

    logging.info("\n--- Test Complete ---")
    logging.info("You can now exit the container and review the two PNG files.")

if __name__ == "__main__":
    main()
