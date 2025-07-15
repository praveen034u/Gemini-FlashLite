from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gemini_langchain import GeminiLLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

app = FastAPI()

# Store memory per user
user_memories = {}

class PromptInput(BaseModel):
    user_id: str
    prompt: str

@app.post("/generate")
async def generate(input_data: PromptInput):
    if not input_data.prompt or not input_data.user_id:
        raise HTTPException(status_code=400, detail="user_id and prompt are required")

    try:
        # Get or create memory for the user
        if input_data.user_id not in user_memories:
            user_memories[input_data.user_id] = ConversationBufferMemory(return_messages=True)

        memory = user_memories[input_data.user_id]
        
        # Gemini as a LangChain-compatible LLM
        llm = GeminiLLM()

        # Create a conversation chain with memory
        chain = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True
        )

        response = chain.run(input_data.prompt)

        return {
            "response": response,
            "conversation_history": memory.buffer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
