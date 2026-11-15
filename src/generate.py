"""Step 6: ask Gemini to answer using ONLY the retrieved chunks (with retry on 429/5xx)."""
import os
import time

from google import genai
from google.genai import errors, types

from src import config

SYSTEM = (
    "You answer questions using ONLY the numbered context passages provided. "
    "If the context does not contain the answer, say you don't know. "
    "Cite sources inline like [1], [2] matching the passage numbers. Be concise."
)
MAX_ATTEMPTS = 4


def build_context(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        page = f", p.{c['page']}" if c.get("page") else ""
        lines.append(f"[{i}] ({c['source']}{page})\n{c['text']}")
    return "\n\n".join(lines)


def generate_answer(query: str, chunks: list[dict]) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY. Add it to your .env file.")
    client = genai.Client(api_key=api_key)
    prompt = f"Context:\n{build_context(chunks)}\n\nQuestion: {query}"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=config.LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    temperature=0.2,
                    max_output_tokens=2048,  # newer models spend some tokens "thinking"
                ),
            )
            return response.text or "(The model returned no text. Try again.)"
        except errors.APIError as e:
            if e.code in (429, 500, 503) and attempt < MAX_ATTEMPTS:
                wait = 5 * 2 ** (attempt - 1)
                print(f"[Gemini busy/rate-limited ({e.code}). Retrying in {wait}s...]")
                time.sleep(wait)
                continue
            raise
