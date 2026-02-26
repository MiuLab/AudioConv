import os
from time import sleep
from typing import Union, List, Tuple, Optional
from google import genai
from google.genai import types
from audioconv.llms.utils import retry_with_exponential_backoff

class Gemini:
    """Wrapper class for Google's Gemini API with support for text and audio inputs."""

    SAFETY_SETTINGS = [types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_IMAGE_HATE",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_IMAGE_HARASSMENT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT",
        threshold="OFF"
        )
    ]
    def __init__(self, model_name: str = 'gemini-1.5-pro', api_key: Optional[str] = None) -> None:
        """
        Initialize the Gemini model.

        Args:
            model_name: The name of the Gemini model to use
            api_key: Optional API key. If not provided, will look for GEMINI_API_KEY in environment
        """
        if api_key is None:
            api_key = os.environ["GEMINI_API_KEY"]

        self.generation_config = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_output_tokens": 1024,
        }
        self.client = genai.Client(
            vertexai=True,
            project=os.environ['GCP_PROJECT_NAME'],
            location="global",
        )

        self.model_name = model_name


    def __str__(self) -> str:
        return self.model_name

    @retry_with_exponential_backoff
    def __call__(
        self,
        conversations,
        max_tokens: int = 8192,
        top_p: float = 1.0,
        top_k: int = 1,
        temperature: float = 0.0,
        thinking_budget=8192,
        **kwargs
    ) -> Tuple[str, dict]:
        """
        Generate content using the Gemini model.

        Args:
            conversations: List of conversation turns
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            temperature: Temperature for sampling
            **kwargs: Additional arguments (response_mime_type, response_schema)

        Returns:
            Generated text
        """

        convo = []
        for conversation in conversations:
            parts = []
            for part in conversation['parts']:
                if part['type'] == 'audio':
                    # Infer MIME type from file extension (default to wav)
                    path = part['value']
                    ext = os.path.splitext(path)[1].lower()
                    if ext in ('.mp3', '.mpeg'):
                        mime = "audio/mpeg"
                    elif ext in ('.wav',):
                        mime = "audio/wav"
                    elif ext in ('.flac',):
                        mime = "audio/flac"
                    elif ext in ('.m4a', '.aac'):
                        mime = "audio/mp4"
                    else:
                        mime = "audio/wav"
                    with open(path, 'rb') as f:
                        g_part = types.Part.from_bytes(
                            data=f.read(),
                            mime_type=mime,
                        )
                elif part['type'] == 'text':
                    g_part = types.Part.from_text(text=part['value'])
                parts.append(g_part)
            role = conversation['role']
            if role == 'system':
                role = 'user'
            convo.append(types.Content(
                role=role,
                parts=parts
            ))
        # Optional JSON/structured response controls
        response_mime_type = kwargs.get("response_mime_type")
        response_schema = kwargs.get("response_schema")

        config_kwargs = dict(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            response_modalities=["TEXT"],
            safety_settings=self.SAFETY_SETTINGS,
        )
        if response_mime_type is not None:
            config_kwargs["response_mime_type"] = response_mime_type
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema
        generate_content_config = types.GenerateContentConfig(**config_kwargs)
        res = self.client.models.generate_content(
            model = self.model_name,
            contents = convo,
            config = generate_content_config,

        )
        response_text = res.candidates[0].content.parts[0].text
        return response_text
