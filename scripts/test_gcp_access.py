import vertexai
from vertexai.generative_models import GenerativeModel # --- THIS IS THE FIX ---

PROJECT_ID = "tara-lora-project"
LOCATION = "us-central1"

print(f"Attempting to initialize for project: {PROJECT_ID} in {LOCATION}...")
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print("Initialization successful.")
except Exception as e:
    print(f"\n--- FAILED TO INITIALIZE ---")
    print(f"Error: {e}")
    print("This is the source of the problem. Check project/location/auth.")
    exit()

print("\nAttempting to list available models...")
try:
    # Use the same GenerativeModel.list() call as before
    models = GenerativeModel.list()
    
    print("\n--- SUCCESS ---")
    print("Found the following models:")
    found_one = False
    for model in models:
        # Print just the model IDs we care about
        if 'gemini' in model.name:
            print(model.name)
            found_one = True
    
    if not found_one:
        print("No Gemini models found in this region.")
            
except Exception as e:
    print(f"\n--- FAILED TO LIST MODELS ---")
    print(f"Error: {e}")
    print("\nThis confirms your project has a policy block.")
    print("You are authenticated, but your organization's security policies")
    print("are preventing you from seeing or using any models.")
