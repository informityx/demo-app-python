from openai import OpenAI
import os
from app.config import Config


def get_openai_client() -> OpenAI:
    """Get OpenAI client instance"""
    api_key = Config.OPENAI_API_KEY

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in secrets.toml or as an environment variable."
        )

    # Bound stall time: default SDK can wait a very long time on bad networks.
    timeout = float(os.getenv("OPENAI_TIMEOUT_SEC", "120"))

    return OpenAI(api_key=api_key, timeout=timeout)
