
from langchain_core.language_models import LLM
from typing import Optional, List
from google import genai
from google.genai import types

class GeminiLLM(LLM):
    model_name: str = "gemini-2.5-flash-lite-preview-06-17"
    project: str = "numeric-advice-463218-r7"
    location: str = "global"

    @property
    def _llm_type(self) -> str:
        return "vertexai-genai-stream"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = genai.Client(
            vertexai=True,
            project=self.project,
            location=self.location,
        )

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt)
                ]
            )
        ]

        config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=2048,
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
