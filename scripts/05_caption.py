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
MODEL_ID = "gemini-2.0-flash-001" # The correct, active model

# --- NEW: The "Scalpel" Prompt ---
# This is the fix that satisfies both of your constraints.
API_PROMPT = """You are a captioning expert for AI model training.
Your task is to describe ONLY THE PERSON in the image.
Focus on their clothing, pose, action, and expression.
DO NOT describe the background. Ignore it completely.
Do not mention tattoos, watermarks, or text.
Be concise and factual. Do not use the subject's name.
Example: 'long brown hair, wearing a pink cardigan, looking over their shoulder'"""

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
}
# --- End Configuration ---

def main():
    parser = argparse.ArgumentParser(description="Caption images using Vertex AI.")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        required=True, 
        help="The base project directory (e.g., /projects/tara_tainton_TEST)"
    )
    parser.add_argument(
        "--project_id", 
        type=str, 
        required=True, 
        help="Your Google Cloud Project ID"
    )
    parser.add_argument(
        "--trigger_words", 
        type=str, 
        default="ohwx tara, ", 
        help="Trigger words to prefix to every caption"
    )
    args = parser.parse_args()
    
    INPUT_DIR = Path(args.base_dir) / "05_training_data"

    print(f"--- Starting Step 6: VERTEX AI Captioning (Background-Ignored) ---")
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

    source_folders = [f for f in INPUT_DIR.iterdir() if f.is_dir()]
    if not source_folders:
        print(f"Error: No category folders found in {INPUT_DIR}. Exiting.")
        return

    for folder_path in source_folders:
        print(f"\nProcessing category: {folder_path.name}")
        
        image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg")) + \
                      list(folder_path.glob("*.png")) + list(folder_path.glob("*.webp"))
        
        if not image_files:
            print(f"No images found in {folder_path}. Skipping.")
            continue
            
        print(f"Found {len(image_files)} images. Starting caption generation...")

        for img_path in tqdm(image_files, desc=f"Generating captions in {folder_path.name}"):
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
                
                # Gender-neutral replacements
                caption = caption.replace("a woman ", "", 1).replace("a photo of a woman ", "", 1)
                caption = caption.replace("a man ", "", 1).replace("a photo of a man ", "", 1)
                caption = caption.replace("a person ", "", 1).replace("a photo of a person ", "", 1)
                caption = caption.replace("photo of ", "", 1).replace("image of ", "", 1)

                # --- Hard-filter for tattoo/watermark ---
                caption = caption.replace("tattoo", "").replace("tattoos", "")
                caption = caption.replace(", ,", ",").replace("  ", " ").replace(" ,", ",").strip()

                final_caption = args.trigger_words + caption
                txt_path.write_text(final_caption)
                
                time.sleep(1.1) 
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                if "quota" in str(e).lower():
                    print("Quota error. Pausing for 60 seconds..."); time.sleep(60)
                    
    print("\n--- Process Complete ---")
    print(f"Captioning finished for: {INPUT_DIR}")

if __name__ == "__main__":
    main()
