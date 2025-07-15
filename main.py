from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gemini_langchain import GeminiLLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine, text
import os
import uuid
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Starting FastAPI app")

app = FastAPI()

# Supabase PostgreSQL connection string (set in environment)
POSTGRES_URL = os.getenv("POSTGRES_URL")  # Format: postgresql+psycopg2://user:password@host:5432/dbname
engine = create_engine(POSTGRES_URL)

# Input model
class PromptInput(BaseModel):
    user_id: str
    prompt: str

@app.post("/generate")
async def generate(input_data: PromptInput):
    if not input_data.prompt or not input_data.user_id:
        raise HTTPException(status_code=400, detail="user_id and prompt are required")

    try:
        # Get latest session and message count
        with engine.connect() as conn:
            result = conn.execute(text("""
            SELECT session_id, COUNT(*) AS message_count
            FROM message_store
            WHERE session_id LIKE :user_id_prefix
            GROUP BY session_id
            ORDER BY MAX(created_at) DESC
            LIMIT 1
            """), {"user_id_prefix": f"{input_data.user_id}%"})

        row = result.fetchone()

        if row:
         row = row._mapping

        if not row or row['message_count'] >= 10:
         session_id = f"{input_data.user_id}_{uuid.uuid4().hex[:8]}"
        else:
         session_id = row['session_id']

        # Use LangChain SQL memory
        chat_history = SQLChatMessageHistory(
            connection_string=POSTGRES_URL,
            session_id=session_id
        )

        memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            chat_memory=chat_history
        )

        # Gemini LLM
        llm = GeminiLLM()
        chain = ConversationChain(llm=llm, memory=memory, verbose=True)

        # Generate response
        response = chain.run(input_data.prompt)

        return {
            "response": response,
            "session_id": session_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{user_id}")
async def get_chat_history(user_id: str):
    try:
        with engine.connect() as conn:
           result = conn.execute(text("""
                SELECT session_id, role, content, created_at AS timestamp
                FROM message_store
                WHERE session_id LIKE :user_id_prefix
                ORDER BY session_id, created_at ASC
                """), {"user_id_prefix": f"{user_id}%"})

        history = [dict(row._mapping) for row in result]

        return {
            "user_id": user_id,
            "chat_history": history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
