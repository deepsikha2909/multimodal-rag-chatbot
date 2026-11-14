
import os
from dotenv import load_dotenv
from google import genai

# Load API key from .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Gemini API key not found. Check your .env file.")

# Create Gemini client
client = genai.Client(api_key=API_KEY)

# Select Gemini model
MODEL_NAME = "gemini-3.5-flash-lite"

