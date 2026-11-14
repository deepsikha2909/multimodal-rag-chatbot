import os
from dotenv import load_dotenv
from google import genai
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

# Load local environment variables for the fallback.
load_dotenv()

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, StreamlitSecretNotFoundError):
    API_KEY = None

if not API_KEY:
    API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "Gemini API key not found. Configure GEMINI_API_KEY in Streamlit "
        "secrets or your environment/.env file."
    )

# Create Gemini client
client = genai.Client(api_key=API_KEY)

# Select Gemini model
MODEL_NAME = "gemini-3.5-flash-lite"