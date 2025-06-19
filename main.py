
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gemini_langchain import GeminiLLM

app = FastAPI()

class PromptInput(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(input_data: PromptInput):
    if not input_data.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        llm = GeminiLLM()
        response = llm._call(input_data.prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
