"""
Shared configuration constants read from environment variables.
"""
import os

OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")
