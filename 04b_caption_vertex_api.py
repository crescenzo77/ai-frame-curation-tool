import sys
import os
from pathlib import Path
from tqdm import tqdm
import base64
import time

# Import the new Vertex AI libraries
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Part,
    Image,
    GenerationConfig,
    HarmCategory,
    SafetySetting,
)

# --- Configuration ---
PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"
MODEL_ID = "gemini-1.5-pro-latest"
IMAGE_FOLDER = Path("/projects/sorted_output/a_face_n_hair_top60_v4")
TRIGGER_PREFIX = "my_trigger_token, "

API_PROMPT = """You are a captioning expert for AI model training.
Describe this image objectively. Focus on the person, clothing, pose, action, and any visible, distinct features.
Be concise and factual. Do not use the subject's name.
Example: 'a person with long brown hair, wearing a pink cardigan, looking over their shoulder'"""

# --- Disable all safety filters ---
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
}

# --- Main Execution ---
def main():
    print(f"Starting VERTEX AI captioning process for: {IMAGE_FOLDER}")

    try:
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # Load the model
        model = GenerativeModel(
            MODEL_ID,
            system_instruction=[API_PROMPT]
        )
        print(f"Successfully loaded Vertex AI model: {MODEL_ID}")

    except Exception as e:
        print(f"Error initializing Vertex AI: {e}")
        print("Check your GOOGLE_APPLICATION_CREDENTIALS path and permissions.")
        return

    # Find all image files
    image_files = list(IMAGE_FOLDER.glob("*.jpg")) + \
                  list(IMAGE_FOLDER.glob("*.jpeg")) + \
                  list(IMAGE_FOLDER.glob("*.png")) + \
                  list(IMAGE_FOLDER.glob("*.webp"))

    if not image_files:
        print(f"Error: No images found in {IMAGE_FOLDER}. Exiting.")
        return

    print(f"Found {len(image_files)} images. Starting caption generation...")

    # Loop through every image
    for img_path in tqdm(image_files, desc="Generating captions (Vertex AI)"):

        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            continue

        try:
            # 1. Load image bytes
            with open(img_path, "rb") as f:
                image_bytes = f.read()

            # 2. Create the prompt parts
            mime_type = "image/jpeg" if img_path.suffix == ".jpg" else f"image/{img_path.suffix.lstrip('.')}"
            image_part = Part.from_data(image_bytes, mime_type=mime_type)

            # 3. Generate content with safety filters OFF
            response = model.generate_content(
                [image_part],
                safety_settings=SAFETY_SETTINGS,
                generation_config=GenerationConfig(temperature=0.2) # Factual
            )

            # 4. Extract and save
            caption = response.text.strip()

            # --- GENDER-NEUTRAL REPLACEMENT ---
            caption = caption.replace("a woman ", "", 1)
            caption = caption.replace("a photo of a woman ", "", 1)
            caption = caption.replace("a man ", "", 1)
            caption = caption.replace("a photo of a man ", "", 1)
            caption = caption.replace("a person ", "", 1)
            caption = caption.replace("a photo of a person ", "", 1)
            # --- END FIX ---

            final_caption = TRIGGER_PREFIX + caption
            txt_path.write_text(final_caption)

            # Vertex has a high quota (e.g., 60/min)
            time.sleep(1.1)

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            if "quota" in str(e).lower():
                print("Quota error. Pausing for 60 seconds...")
                time.sleep(60)

    print("\n--- Process Complete ---")
    print(f"All images in {IMAGE_FOLDER} have been captioned with the Vertex AI API.")

if __name__ == "__main__":
    main()
