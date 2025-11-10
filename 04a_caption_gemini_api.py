import sys
import os
import google.genai as genai
from pathlib import Path
from tqdm import tqdm
import PIL.Image
import time

# --- Configuration ---
API_KEY = "YOUR_API_KEY_GOES_HERE"
MODEL_ID = "gemini-1.5-flash-latest"
IMAGE_FOLDER = Path("/projects/sorted_output/a_face_n_hair_top60_v4")
TRIGGER_PREFIX = "my_trigger_token, "

API_PROMPT = """You are a captioning expert for AI model training.
Describe this image objectively. Focus on the person, clothing, pose, action, and any visible, distinct features.
Be concise and factual. Do not use the subject's name.
Example: 'a person with long brown hair, wearing a pink cardigan, looking over their shoulder'"""

# --- Main Execution ---
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
        print("Please check your API_KEY and internet connection.")
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
    for img_path in tqdm(image_files, desc="Generating captions (Cloud API)"):
        
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            continue
            
        try:
            image = PIL.Image.open(img_path)
            
            response = model.generate_content(
                contents=[image, API_PROMPT] # Use 'contents' (plural)
            )
            
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
            
            # API Rate Limit Delay
            time.sleep(1.1)
            
        except Exception as e:
            if "API_KEY_INVALID" in str(e) or "API key not valid" in str(e):
                print("\n\n*** FATAL ERROR: Your API Key is invalid. ***")
                break
            if "quota" in str(e).lower():
                print(f"\nQuota error processing {img_path}: {e}")
                print("Pausing for 60 seconds before retrying...")
                time.sleep(60)
                # Retry this image
                try:
                    image = PIL.Image.open(img_path)
                    response = model.generate_content(
                        contents=[image, API_PROMPT]
                    )
                    caption = response.text.strip()
                    # --- GENDER-NEUTRAL REPLACEMENT (for retry) ---
                    caption = caption.replace("a woman ", "", 1)
                    caption = caption.replace("a photo of a woman ", "", 1)
                    caption = caption.replace("a man ", "", 1)
                    caption = caption.replace("a photo of a man ", "", 1)
                    caption = caption.replace("a person ", "", 1)
                    caption = caption.replace("a photo of a person ", "", 1)
                    # --- END FIX ---
                    final_caption = TRIGGER_PREFIX + caption
                    txt_path.write_text(final_caption)
                except Exception as e2:
                    print(f"Retry failed for {img_path}: {e2}")
            else:
                print(f"\nError processing {img_path}: {e}")

    print("\n--- Process Complete ---")
    print(f"All images in {IMAGE_FOLDER} have been captioned with the Gemini API.")

if __name__ == "__main__":
    main()
