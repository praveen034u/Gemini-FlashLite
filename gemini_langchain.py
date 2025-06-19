from typing import Any, List, Optional
from langchain_core.language_models import LLM
from langchain_core.outputs import Generation
from google import genai
from google.genai import types

class GeminiLLM(LLM):
    model: str = "gemini-2.5-flash-lite-preview-06-17"
    project: str = "numeric-advice-463218-r7"
    location: str = "global"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = genai.Client(vertexai=True, project=self.project, location=self.location)
        contents = [types.Content(role="user", parts=[types.Part.from_text(prompt)])]

        config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=2048,
            safety_settings=[types.SafetySetting(category=cat, threshold="BLOCK_LOW_AND_ABOVE") for cat in [
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_HARASSMENT"
            ]]
        )

        response = client.models.generate_content(model=self.model, contents=contents, config=config)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini-custom"
