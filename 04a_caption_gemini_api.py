#!/usr/bin/env python3

import sys
import os
import google.genai as genai
from pathlib import Path
from tqdm import tqdm
import PIL.Image
import time

# --- Configuration ---
# !!! IMPORTANT: You must edit this script to set your API_KEY
API_KEY = "YOUR_API_KEY_GOES_HERE"
MODEL_ID = "gemini-1.5-flash-latest"

# !!! IMPORTANT: You must edit this to match your desired final folder
IMAGE_FOLDER = Path("/projects/sorted_output/a_face_n_hair_top60_v4_temporal")
TRIGGER_PREFIX = "my_trigger_token, "

API_PROMPT = """You are a captioning expert for AI model training.
Describe this image objectively. Focus on the person, clothing, pose, action, and any visible, distinct features.
Be concise and factual. Do not use the subject's name.
Example: 'a person with long brown hair, wearing a pink cardigan, looking over their shoulder'"""
# --- End Configuration ---

def main():
    print(f"Starting CLOUD API captioning process for: {IMAGE_FOLDER}")
    if API_KEY == "YOUR_API_KEY_GOES_HERE":
        print("\n*** FATAL ERROR: You must edit the script and paste your API_KEY. ***")
        return
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(MODEL_ID)
        print(f"Successfully configured Gemini API model: {MODEL_ID}")
    except Exception as e:
        print(f"Error configuring API: {e}")
        return

    image_files = list(IMAGE_FOLDER.glob("*.jpg")) + list(IMAGE_FOLDER.glob("*.jpeg")) + \
                  list(IMAGE_FOLDER.glob("*.png")) + list(IMAGE_FOLDER.glob("*.webp"))
    if not image_files:
        print(f"Error: No images found in {IMAGE_FOLDER}. Exiting.")
        return
    print(f"Found {len(image_files)} images. Starting caption generation...")

    for img_path in tqdm(image_files, desc="Generating captions (Cloud API)"):
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            continue
        try:
            image = PIL.Image.open(img_path)
            response = model.generate_content(contents=[image, API_PROMPT])
            caption = response.text.strip()
            
            # Gender-neutral replacements
            caption = caption.replace("a woman ", "", 1).replace("a photo of a woman ", "", 1)
            caption = caption.replace("a man ", "", 1).replace("a photo of a man ", "", 1)
            caption = caption.replace("a person ", "", 1).replace("a photo of a person ", "", 1)
            
            final_caption = TRIGGER_PREFIX + caption
            txt_path.write_text(final_caption)
            time.sleep(1.1) # Adhere to standard Gemini API rate limits
        except Exception as e:
            if "API_KEY_INVALID" in str(e) or "API key not valid" in str(e):
                print("\n\n*** FATAL ERROR: Your API Key is invalid. ***"); break
            if "quota" in str(e).lower():
                print(f"\nQuota error on {img_path}. Pausing 60s..."); time.sleep(60)
                try: # Retry once
                    image = PIL.Image.open(img_path)
                    response = model.generate_content(contents=[image, API_PROMPT])
                    caption = response.text.strip()
                    caption = caption.replace("a woman ", "", 1).replace("a photo of a woman ", "", 1)
                    caption = caption.replace("a man ", "", 1).replace("a photo of a man ", "", 1)
                    caption = caption.replace("a person ", "", 1).replace("a photo of a person ", "", 1)
                    final_caption = TRIGGER_PREFIX + caption
                    txt_path.write_text(final_caption)
                except Exception as e2:
                    print(f"Retry failed for {img_path}: {e2}")
            else:
                print(f"\nError processing {img_path}: {e}")
    print("\n--- Process Complete ---")

if __name__ == "__main__":
    main()
