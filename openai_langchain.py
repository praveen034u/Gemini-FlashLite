try:
    from langchain_core.language_models import LLM
except ImportError:
    from langchain.llms.base import LLM
from typing import Optional, List
from openai import OpenAI
import os


class OpenAILLM(LLM):
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")

    @property
    def _llm_type(self) -> str:
        return "openai"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = OpenAI(api_key=self.api_key)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.95,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
