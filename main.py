from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gemini_langchain import GeminiLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


# memory = ConversationBufferMemory(memory_key="history", return_messages=False)

# prompt_template = PromptTemplate(
#     input_variables=["history", "PromptInput"],
#     template="""
#     The following is a conversation between a user and an assistant.
#     {history}
#     User: {PromptInput}
#     Assistant:"""
# )

# llm = GeminiLLM()
# conversation = LLMChain(
#     llm = llm,
#     prompt = prompt_template,
#     memory = memory,
#     verbose = True
# )

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
