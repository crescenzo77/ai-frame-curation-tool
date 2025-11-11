#!/usr/bin/env python3

import sys, os, time
from pathlib import Path
from tqdm import tqdm
import vertexai
from vertexai.generative_models import (
    GenerativeModel, Part, GenerationConfig, HarmCategory, SafetySetting
)

# --- Configuration ---
# !!! IMPORTANT: You must edit this to set your GCP Project ID
PROJECT_ID = "your-gcp-project-id" 
LOCATION = "us-central1"
MODEL_ID = "gemini-1.5-pro-latest"

# !!! IMPORTANT: You must edit this to match your desired final folder
IMAGE_FOLDER = Path("/projects/sorted_output/a_face_n_hair_top60_v4_temporal")
TRIGGER_PREFIX = "my_trigger_token, "

API_PROMPT = """You are a captioning expert for AI model training.
Describe this image objectively. Focus on the person, clothing, pose, action, and any visible, distinct features.
Be concise and factual. Do not use the subject's name.
Example: 'a person with long brown hair, wearing a pink cardigan, looking over their shoulder'"""

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
}
# --- End Configuration ---

def main():
    print(f"Starting VERTEX AI captioning process for: {IMAGE_FOLDER}")
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_ID, system_instruction=[API_PROMPT])
        print(f"Successfully loaded Vertex AI model: {MODEL_ID}")
    except Exception as e:
        print(f"Error initializing Vertex AI: {e}")
        print("Check your GOOGLE_APPLICATION_CREDENTIALS path and permissions.")
        return

    image_files = list(IMAGE_FOLDER.glob("*.jpg")) + list(IMAGE_FOLDER.glob("*.jpeg")) + \
                  list(IMAGE_FOLDER.glob("*.png")) + list(IMAGE_FOLDER.glob("*.webp"))
    if not image_files:
        print(f"Error: No images found in {IMAGE_FOLDER}. Exiting.")
        return
    print(f"Found {len(image_files)} images. Starting caption generation...")

    for img_path in tqdm(image_files, desc="Generating captions (Vertex AI)"):
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            continue
        try:
            with open(img_path, "rb") as f: image_bytes = f.read()
            mime_type = "image/jpeg" if img_path.suffix == ".jpg" else f"image/{img_path.suffix.lstrip('.')}"
            image_part = Part.from_data(image_bytes, mime_type=mime_type)

            response = model.generate_content(
                [image_part],
                safety_settings=SAFETY_SETTINGS,
                generation_config=GenerationConfig(temperature=0.2) # Factual
            )
            caption = response.text.strip()
            
            # Gender-neutral replacements
            caption = caption.replace("a woman ", "", 1).replace("a photo of a woman ", "", 1)
            caption = caption.replace("a man ", "", 1).replace("a photo of a man ", "", 1)
            caption = caption.replace("a person ", "", 1).replace("a photo of a person ", "", 1)

            final_caption = TRIGGER_PREFIX + caption
            txt_path.write_text(final_caption)
            time.sleep(1.1) # Adhere to Vertex AI quotas (60/min)
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            if "quota" in str(e).lower():
                print("Quota error. Pausing for 60 seconds..."); time.sleep(60)
    print("\n--- Process Complete ---")

if __name__ == "__main__":
    main()
