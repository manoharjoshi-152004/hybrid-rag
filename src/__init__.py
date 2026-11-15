import os

# Silences the harmless Hugging Face symlink warning on Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
