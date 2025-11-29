"""
Hugging Face Authentication Setup
This script logs in to Hugging Face using an API token stored in .env file
"""

import os
from dotenv import load_dotenv
from huggingface_hub import login

def authenticate_huggingface():
    """Authenticate with HuggingFace"""
    # Load environment variables from .env file
    load_dotenv()

    # Get Hugging Face token from environment
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN not found in .env file. Please add your token.")

    # Login to Hugging Face
    login(hf_token)
    print("✓ Successfully logged in to Hugging Face!")
    return hf_token

if __name__ == "__main__":
    authenticate_huggingface()
