"""
llm_client.py — LLM Client Factory for Smart Nutrition Tracker.
Supports Gemini (default) and OpenAI. Matches ExVenture multi-agent pattern.

Usage:
    from llm_client import get_llm_client, call_llm, call_llm_with_json

    # Factory pattern (matching old project)
    client, model = get_llm_client("gemini")
    response = client.chat.completions.create(model=model, messages=[...])

    # High-level helpers
    text, duration = call_llm("What food is this?", system_prompt="...")
    json_text, duration = call_llm_with_json("Parse this food: pizza")
"""

import os
import time
from typing import Optional
from openai import OpenAI
from config import GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL, LLM_SEED


# ---------------------------------------------------------------------------
# LLM client factory — supports Gemini and OpenAI
# ---------------------------------------------------------------------------

def get_llm_client(provider: str = "gemini", model: str = None):
    """
    Returns (client, model_name) for the requested provider.

    provider: 'gemini' | 'openai'
    model:    override the default model name (optional)

    Environment variables:
        GEMINI_API_KEY  — required for Gemini (default)
        OPENAI_API_KEY  — required for OpenAI
    """
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        client = OpenAI(api_key=api_key)
        model_name = model or "gpt-4o-mini"
    else:  # gemini (default)
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not set. Export it or add to .env file.\n"
                "  export GEMINI_API_KEY=your_key_here"
            )
        client = OpenAI(
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_BASE_URL,
        )
        model_name = model or GEMINI_MODEL

    return client, model_name


# ---------------------------------------------------------------------------
# High-level helpers — call_llm / call_llm_with_json
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    provider: str = "gemini",
    model_name: str = None,
    seed: Optional[int] = None,
) -> tuple:
    """
    Send a prompt to LLM and return (response_text, duration_seconds).
    Supports provider switching via payload.
    Uses LLM_SEED from config for reproducibility (override with seed param).
    """
    client, model = get_llm_client(provider, model_name)
    effective_seed = seed if seed is not None else LLM_SEED

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        # seed=effective_seed,  # Gemini API does not support seed param
    )
    duration = time.time() - start

    text = response.choices[0].message.content.strip()
    return text, duration


def call_llm_with_json(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    provider: str = "gemini",
    model_name: str = None,
) -> tuple:
    """
    Same as call_llm but instructs the model to respond in JSON only.
    Lower default temperature for structured output.
    """
    json_instruction = (
        "You MUST respond with valid JSON only. "
        "No markdown, no explanation, no code fences. Just the JSON object."
    )
    if system_prompt:
        full_system = f"{system_prompt}\n\n{json_instruction}"
    else:
        full_system = json_instruction

    return call_llm(prompt, full_system, temperature, max_tokens, provider, model_name)


# ---------------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing LLM client...\n")

    text, dur = call_llm(
        prompt="What is 2 + 2? Answer in one word.",
        system_prompt="You are a helpful assistant. Be brief.",
    )
    print(f"Response: {text}")
    print(f"Duration: {dur:.2f}s")

    print("\nTesting JSON mode...")
    json_text, dur = call_llm_with_json(
        prompt="List 3 fruits with their calories per 100g.",
        system_prompt="You are a nutrition database.",
    )
    print(f"JSON Response: {json_text}")
    print(f"Duration: {dur:.2f}s")

    print("\nLLM client works!")
