from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import json
from gemini_langchain import GeminiLLM
from typing import Dict
import openai

app = FastAPI()

# Load LLM config
def load_llm_config():
    with open('llm_config.json', 'r') as f:
        return json.load(f)

llm_config = load_llm_config()

# Request model
class LLMRequest(BaseModel):
    role: str  # 'admin', 'teacher', 'student'
    llm: Optional[str] = None  # LLM name from config
    prompt: str



# Generic LangChain LLM handler (extend for more LLMs as needed)

# OpenAI LLM handler
class OpenAILLM:
    def __init__(self, config: Dict):
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2048)
        # Optionally set your OpenAI API key here or via env var

    def call(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message["content"].strip()

# Azure OpenAI LLM handler
class AzureOpenAILLM:
    def __init__(self, config: Dict):
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2048)
        # Set Azure OpenAI endpoint and key via env vars or config

    def call(self, prompt: str) -> str:
        # This assumes openai-python is configured for Azure endpoint
        response = openai.ChatCompletion.create(
            engine=self.model,  # For Azure, 'engine' is used instead of 'model'
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message["content"].strip()

class LangChainLLM:
    def __init__(self, llm_name: str, config: Dict):
        self.llm_name = llm_name.upper()
        self.config = config
        if self.llm_name == "GEMINI":
            self.llm = GeminiLLM()
        elif self.llm_name == "OPEN-API":
            self.llm = OpenAILLM(config)
        elif self.llm_name == "AZURE-OPENAPI":
            self.llm = AzureOpenAILLM(config)
        else:
            self.llm = None

    def call(self, prompt: str) -> str:
        if self.llm:
            return self.llm.call(prompt) if hasattr(self.llm, "call") else self.llm._call(prompt)
        return f"[Simulated response from {self.llm_name} for prompt: {prompt}]"

def call_llm(llm_name, prompt, system_prompt=None):
    sys_prompt = system_prompt or ""
    full_prompt = f"{sys_prompt}\n\n{prompt}" if sys_prompt else prompt
    llm_config_entry = llm_config['llms'].get(llm_name)
    lc_llm = LangChainLLM(llm_name, llm_config_entry)
    response = lc_llm.call(full_prompt)
    return {
        "llm": llm_name,
        "system_prompt": sys_prompt,
        "response": response
    }

@app.post("/api/llm")
async def llm_api(request: LLMRequest):
    role = request.role.lower()
    llm_name = request.llm or llm_config.get('default_llm')
    prompt = request.prompt

    # Role-based logic (expand as needed)
    if role not in ['admin', 'teacher', 'student']:
        return {"error": "Invalid role"}

    # Check if LLM exists
    if llm_name not in llm_config['llms']:
        return {"error": f"LLM '{llm_name}' not found in config"}

    # Get system prompt for the role, fallback to default
    system_prompts = llm_config.get('system_prompts', {})
    system_prompt = system_prompts.get(role, system_prompts.get('default', ""))

    # Call the LLM (replace with real call)
    result = call_llm(llm_name, prompt, system_prompt)
    return {
        "role": role,
        "llm": llm_name,
        "system_prompt": system_prompt,
        "result": result
    }

@app.get("/")
def root():
    return {"message": "BrightMinds LLM Wrapper API"}
