
try:
    from langchain_core.language_models import LLM
except ImportError:
    from langchain.llms.base import LLM
from typing import Optional, List
import google.genai as genai
from google.genai import types
import os

class GeminiLLM(LLM):
    model_name: str = "gemini-2.5-flash-lite-preview-06-17"
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.api_key:
            self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Please check your .env file.")

    @property
    def _llm_type(self) -> str:
        return "google-genai"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = genai.Client(api_key=self.api_key)

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt)
                ]
            )
        ]

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=0.95,
            max_output_tokens=self.max_tokens,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
            ],
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        # Stream and accumulate the result
        result = ""
        for chunk in client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config,
        ):
            if hasattr(chunk, "text") and chunk.text:
                result += chunk.text
        return result
