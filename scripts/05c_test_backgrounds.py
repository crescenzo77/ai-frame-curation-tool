#!/usr/bin/env python3

import sys, os, time
from pathlib import Path
from tqdm import tqdm
import vertexai
from vertexai.generative_models import (
    GenerativeModel, Part, GenerationConfig, HarmCategory, SafetySetting
)
import argparse
import logging

# --- Configuration ---
LOCATION = "us-central1"
MODEL_ID = "gemini-2.0-flash-001"

# --- Use the NEW Anti-Tattoo/Watermark Prompt ---
# We ask it to describe what it sees. If it's a "cityscape" or "mountain",
# it will tell us. If it's just a blur, it will say "abstract pattern".
API_PROMPT = """You are a scene description expert.
Describe this image objectively and concisely.
Do not mention tattoos, watermarks, or text."""

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
}
# --- End Configuration ---

def main():
    # --- FIXED: Corrected argument parser ---
    parser = argparse.ArgumentParser(description="Unit test backgrounds for caption noise.")
    parser.add_argument(
        "--project_id", 
        type=str, 
        required=True, 
        help="Your Google Cloud Project ID"
    )
    args = parser.parse_args()
    
    # --- FIXED: Point directly to the main background library ---
    INPUT_DIR = Path("/projects/00_background_library")

    print(f"--- Starting Background Unit Test ---")
    print(f"Targeting: {INPUT_DIR}")
    
    if not INPUT_DIR.is_dir():
        print(f"*** FATAL ERROR: Input dir not found: {INPUT_DIR} ***")
        return

    try:
        vertexai.init(project=args.project_id, location=LOCATION)
        model = GenerativeModel(MODEL_ID, system_instruction=[API_PROMPT])
        print(f"Successfully loaded Vertex AI model: {MODEL_ID} for project {args.project_id}")
    except Exception as e:
        print(f"Error initializing Vertex AI: {e}")
        return

    # --- FIXED: No sub-folders, just image files ---
    image_files = list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.jpeg")) + \
                  list(INPUT_DIR.glob("*.png")) + list(INPUT_DIR.glob("*.webp"))
    
    if not image_files:
        print(f"Error: No images found in {INPUT_DIR}. Exiting.")
        return
        
    print(f"Found {len(image_files)} backgrounds. Starting caption generation...")

    for img_path in tqdm(image_files, desc="Testing backgrounds"):
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            tqdm.write(f"Skipping {img_path.name}, caption already exists.")
            continue
        try:
            with open(img_path, "rb") as f: image_bytes = f.read()
            mime_type = "image/jpeg" if img_path.suffix.lower() == ".jpg" else f"image/{img_path.suffix.lstrip('.')}"
            image_part = Part.from_data(image_bytes, mime_type=mime_type)

            response = model.generate_content(
                [image_part],
                safety_settings=SAFETY_SETTINGS,
                generation_config=GenerationConfig(temperature=0.2)
            )
            caption = response.text.strip().lower()
            
            # --- FIXED: No trigger words or replacements needed ---
            txt_path.write_text(caption)
            
            time.sleep(1.1) 
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            if "quota" in str(e).lower():
                print("Quota error. Pausing for 60 seconds..."); time.sleep(60)
                
    print("\n--- Background Test Complete ---")
    print(f"All backgrounds in {INPUT_DIR} have been captioned.")

if __name__ == "__main__":
    main()
