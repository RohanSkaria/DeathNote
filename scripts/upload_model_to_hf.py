"""
Upload model to Hugging Face Hub for Streamlit Cloud deployment
Run this script to upload best_model_state.bin to Hugging Face Hub
"""

from huggingface_hub import HfApi, upload_file
import os

# Configuration
MODEL_PATH = "best_model_state.bin"
REPO_ID = "RohanSkaria/protein-safety-classifier"  # Change to your HF username/repo
HF_TOKEN = os.getenv("HF_TOKEN")  # Set this in your environment

def upload_model():
    """Upload model checkpoint to Hugging Face Hub"""
    if not os.path.exists(MODEL_PATH):
        # Try alternative path
        alt_path = f"visualize-graph-v1/{MODEL_PATH}"
        if os.path.exists(alt_path):
            MODEL_PATH = alt_path
        else:
            print(f"❌ Model file not found: {MODEL_PATH}")
            return False
    
    print(f"📤 Uploading {MODEL_PATH} to Hugging Face Hub...")
    print(f"   Repository: {REPO_ID}")
    
    api = HfApi(token=HF_TOKEN)
    
    try:
        # Upload the model file
        upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo=MODEL_PATH,
            repo_id=REPO_ID,
            repo_type="model",
            token=HF_TOKEN
        )
        print(f"✅ Successfully uploaded to: https://huggingface.co/{REPO_ID}")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

if __name__ == "__main__":
    if not HF_TOKEN:
        print("❌ HF_TOKEN environment variable not set.")
        print("   Set it with: export HF_TOKEN='your_token_here'")
        print("   Get token from: https://huggingface.co/settings/tokens")
    else:
        upload_model()

