import time
import openai
import google.api_core.exceptions as g_exceptions
import urllib.request


def get_llm(model_name: str, series: str = None):
    """
    Factory function to get an LLM instance.

    Args:
        model_name: Model name (e.g., 'gpt-4o-mini', 'gemini-2.5-flash')
        series: API provider ('openai', 'gemini'). Auto-detected if None.

    Returns:
        LLM instance
    """
    if series is None:
        if model_name[:3] in ('gpt', 'o1-'):
            series = 'openai'
        elif model_name[:6] == 'gemini':
            series = 'gemini'
        if series is None:
            raise ValueError(
                f'Unable to auto-detect series for model "{model_name}". '
                'Please provide series explicitly (e.g., series="openai" or series="gemini").'
            )
    if series == 'gemini':
        from audioconv.llms.gemini import Gemini
        return Gemini(model_name)
    elif series == "openai":
        from openai import OpenAI
        import os
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Return a callable wrapper
        return _OpenAIWrapper(client, model_name)
    elif series == "mimo":
        from audioconv.llms.mimo import MiMoAudioChat
        return MiMoAudioChat(model_id=model_name)
    raise ValueError(f'series "{series}" for model "{model_name}" is not supported')


class _OpenAIWrapper:
    """Thin callable wrapper around OpenAI client for use in evaluation scripts."""

    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

    def __str__(self):
        return self.model_name

    def __call__(self, conversations, max_tokens: int = 4096, temperature: float = 0.0, **kwargs):
        messages = []
        for turn in conversations:
            role = turn.get("role", "user")
            parts = turn.get("parts", [])
            text = " ".join(p["value"] for p in parts if p.get("type") == "text")
            messages.append({"role": role, "content": text})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 0.25,
    exponential_base: float = 2,
    max_retries: int = 3
):
    """Retry a function with exponential backoff on API errors."""
    errors_tuple = (
        openai.RateLimitError,
        openai.APIError,
        g_exceptions.ResourceExhausted,
        g_exceptions.ServiceUnavailable,
        g_exceptions.GoogleAPIError,
        urllib.error.HTTPError,
        urllib.error.URLError,
        TypeError,
        ValueError, IndexError, UnboundLocalError
    )

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)
            except errors_tuple as e:
                num_retries += 1
                if isinstance(e, ValueError) or (num_retries > max_retries):
                    result = 'error:{}'.format(e)
                    return result
                print(f"Error ({e}). Retry ({num_retries}) after {delay:.1f}s...")
                time.sleep(delay)
                delay *= exponential_base
            except Exception as e:
                raise e
    return wrapper
