import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from gemini_langchain import GeminiLLM

app = FastAPI()

class PromptInput(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(input_data: PromptInput):
    if not input_data.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    prompt_template = PromptTemplate.from_template(
        "You are a medical assistant. Answer the following question:\n{question}"
    )
    llm = GeminiLLM()
    chain = prompt_template | llm

    response = chain.invoke({"question": input_data.prompt})
    return {"response": response}
