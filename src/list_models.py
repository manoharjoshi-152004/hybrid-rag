"""Helper: print the Gemini models your API key can use.  Run: python -m src.list_models"""
import os

from google import genai

from src import config  # noqa: F401  (loads .env)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if "generateContent" in actions:
        print(m.name.replace("models/", ""))
