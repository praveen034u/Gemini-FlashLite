from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gemini_langchain import GeminiLLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine, text
import os
import uuid
import datetime
import logging

logging.basicConfig(level=logging.INFO)
logging.info("Starting FastAPI app")

app = FastAPI()

POSTGRES_URL = os.getenv("POSTGRES_URL")
engine = create_engine(POSTGRES_URL)

SESSION_EXPIRY_MINUTES = 30

class PromptInput(BaseModel):
    user_id: str
    prompt: str

@app.post("/generate")
async def generate(input_data: PromptInput):
    if not input_data.prompt or not input_data.user_id:
        raise HTTPException(status_code=400, detail="user_id and prompt are required")

    try:
        session_id = None
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT session_id, COUNT(*) AS message_count, MAX(created_at) AS last_activity
                FROM message_store
                WHERE session_id LIKE :user_id_prefix
                GROUP BY session_id
                ORDER BY last_activity DESC
                LIMIT 1
            """), {"user_id_prefix": f"{input_data.user_id}%"})

            row = result.fetchone()
            if row:
                row = row._mapping
                message_count = row['message_count']
                last_activity = row['last_activity']
                if message_count < 10 and (datetime.datetime.utcnow() - last_activity).total_seconds() < SESSION_EXPIRY_MINUTES * 60:
                    session_id = row['session_id']

        if not session_id:
            session_id = f"{input_data.user_id}_{uuid.uuid4().hex[:8]}"

        chat_history = SQLChatMessageHistory(
            connection_string=POSTGRES_URL,
            session_id=session_id
        )

        memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            chat_memory=chat_history
        )

        llm = GeminiLLM()
        chain = ConversationChain(llm=llm, memory=memory, verbose=True)
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
           SELECT session_id,
           message::jsonb->>'type' AS role,
           message::jsonb->'data'->>'content' AS content,
           created_at
           FROM message_store
           WHERE session_id LIKE :user_id_prefix
           ORDER BY session_id, created_at ASC
           """), {"user_id_prefix": f"{user_id}%"})

        messages = [dict(row._mapping) for row in result]

        sessions = {}
        for msg in messages:
            sid = msg['session_id']
            if sid not in sessions:
                sessions[sid] = []
            sessions[sid].append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["created_at"]
            })

        return {
            "user_id": user_id,
            "sessions": sessions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{user_id}")
async def get_user_sessions(user_id: str):
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT DISTINCT session_id, MAX(created_at) AS last_activity
                FROM message_store
                WHERE session_id LIKE :user_id_prefix
                GROUP BY session_id
                ORDER BY last_activity DESC
            """), {"user_id_prefix": f"{user_id}%"})
            sessions = [dict(row._mapping) for row in result]

        return {
            "user_id": user_id,
            "sessions": sessions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))