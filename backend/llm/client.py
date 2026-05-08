
"""
LLM Client — shared across extraction, validation, training, and query generation.

Uses Instructor + OpenAI client pointing at Ollama's local API.
Swap models by changing the model name — works for base and fine-tuned models.

Usage:
    from backend.llm.client import get_client, extract_structured

    # Get a raw instructor client
    client = get_client(model="llama3.2")

    # Or extract structured data directly
    result = extract_structured(
        text="Hans used the Fronius TPS 400i for welding.",
        response_model=Tool,
        model="llama3.2"
    )
"""

from typing import TypeVar, Type
import instructor
from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Ollama runs an OpenAI-compatible API on localhost
OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.2"


def get_openai_client() -> OpenAI:
    """Raw OpenAI client pointing at Ollama."""
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",  # Ollama doesn't need a real key
    )


def get_client() -> instructor.Instructor:
    """Instructor-wrapped client for structured output."""
    return instructor.from_openai(
        get_openai_client(),
        mode=instructor.Mode.JSON,
    )


def extract_structured(
    text: str,
    response_model: Type[T],
    model: str = DEFAULT_MODEL,
    system_prompt: str = "Extract structured information from the given text. Respond in the exact schema requested.",
    temperature: float = 0.0,
    max_retries: int = 3,
) -> T:
    """
    Extract structured data from text using an LLM.
    
    Args:
        text: The input text to extract from
        response_model: Pydantic model class defining the expected output
        model: Ollama model name (base or fine-tuned)
        system_prompt: System instruction for the LLM
        temperature: 0.0 for deterministic extraction
        max_retries: Instructor retries on validation failure
    
    Returns:
        An instance of response_model filled with extracted data
    """
    client = get_client()

    return client.chat.completions.create(
        model=model,
        response_model=response_model,
        max_retries=max_retries,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=temperature,
    )