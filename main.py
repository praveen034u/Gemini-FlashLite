from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from enum import Enum
from gemini_langchain import GeminiLLM
from openai_langchain import OpenAILLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine, text
import os
import uuid
import datetime
import logging
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load LLM configuration from JSON
def load_llm_config():
    config_path = os.path.join(os.path.dirname(__file__), "llm_config.json")
    with open(config_path, "r") as f:
        return json.load(f)

LLM_CONFIG = load_llm_config()

# Define role enum
class RoleEnum(str, Enum):
    teacher = "teacher"
    student = "student"
    admin = "admin"

# LLM handler factory
def get_llm_handler(provider: str, model_name: str, temperature: float, max_tokens: int):
    """Get the appropriate LLM handler based on provider"""
    provider_upper = provider.upper()
    
    if provider_upper == "GEMINI":
        return GeminiLLM(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif provider_upper in ["OPEN-API", "OPENAI"]:
        return OpenAILLM(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif provider_upper == "AZURE-OPENAPI":
        # For now, use OpenAI handler for Azure (can be extended later)
        return OpenAILLM(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")



logging.basicConfig(level=logging.INFO)
logging.info("Starting FastAPI app")

app = FastAPI()

POSTGRES_URL = os.getenv("POSTGRES_URL")
if not POSTGRES_URL:
    raise ValueError("POSTGRES_URL environment variable is not set. Please check your .env file.")

# Remove pgbouncer parameter if present (not supported by psycopg2)
# Supabase uses pgbouncer=true for connection pooling, which needs to be removed
POSTGRES_URL_CLEAN = POSTGRES_URL.replace("?pgbouncer=true", "").replace("&pgbouncer=true", "").replace("?sslmode=require&pgbouncer=true", "?sslmode=require")

engine = create_engine(POSTGRES_URL_CLEAN)

SESSION_EXPIRY_MINUTES = 30

class PromptInput(BaseModel):
    user_id: str
    prompt: str
    role: RoleEnum = RoleEnum.teacher
    llm_provider: str = None

def save_message_history(session_id: str, user_prompt: str, ai_response: str):
    """Background task to save message history to database"""
    try:
        chat_history = SQLChatMessageHistory(
            connection_string=POSTGRES_URL_CLEAN,
            session_id=session_id
        )
        chat_history.add_user_message(user_prompt)
        chat_history.add_ai_message(ai_response)
        logging.info(f"Message history saved for session {session_id}")
    except Exception as e:
        logging.error(f"Error saving message history: {str(e)}")

def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string (rough approximation: 1 token ≈ 4 characters)"""
    return len(text) // 4

def apply_history_limits(messages: list, max_messages: int, max_tokens: int, max_age_days: int) -> list:
    """
    Apply hybrid limits to conversation history:
    - Time-based: Only include messages within max_age_days
    - Count-based: Limit to max_messages (most recent)
    - Token-based: Limit total tokens to max_tokens
    """
    if not messages:
        return []
    
    from datetime import datetime, timedelta
    
    # Filter by age (if messages have timestamp attribute)
    cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)
    filtered_messages = []
    
    for msg in messages:
        # Check if message has timestamp (SQLChatMessageHistory may not have this)
        # If no timestamp, include all messages
        try:
            if hasattr(msg, 'additional_kwargs') and 'timestamp' in msg.additional_kwargs:
                msg_time = datetime.fromisoformat(msg.additional_kwargs['timestamp'])
                if msg_time < cutoff_time:
                    continue
        except:
            pass  # If timestamp parsing fails, include the message
        
        filtered_messages.append(msg)
    
    # Limit by message count (keep most recent N messages)
    if len(filtered_messages) > max_messages:
        filtered_messages = filtered_messages[-max_messages:]
    
    # Limit by token budget (keep most recent messages that fit within token limit)
    token_count = 0
    token_limited_messages = []
    
    # Process messages in reverse order (newest first) to keep most recent context
    for msg in reversed(filtered_messages):
        msg_content = msg.content if hasattr(msg, 'content') else str(msg)
        msg_tokens = estimate_tokens(msg_content)
        
        if token_count + msg_tokens <= max_tokens:
            token_limited_messages.insert(0, msg)  # Insert at beginning to maintain order
            token_count += msg_tokens
        else:
            # Stop adding messages once token limit is reached
            break
    
    logging.info(f"History filtered: {len(messages)} → {len(token_limited_messages)} messages (~{token_count} tokens)")
    return token_limited_messages

@app.post("/generate")
async def generate(input_data: PromptInput, background_tasks: BackgroundTasks):
    if not input_data.prompt or not input_data.user_id:
        raise HTTPException(status_code=400, detail="user_id and prompt are required")

    try:
        # Use default LLM provider if not specified
        if input_data.llm_provider:
            provider = input_data.llm_provider
            logging.info(f"Using user-specified LLM provider: {provider}")
        else:
            provider = LLM_CONFIG.get("default_llm")
            if not provider:
                raise HTTPException(status_code=500, detail="No default_llm configured in llm_config.json")
            logging.info(f"Using default LLM provider from config: {provider}")
        
        provider_upper = provider.upper()
        
        # Validate provider exists in config
        if provider_upper not in LLM_CONFIG.get("llms", {}):
            raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {provider}. Supported: {list(LLM_CONFIG.get('llms', {}).keys())}")
        
        # Get LLM configuration
        llm_config = LLM_CONFIG["llms"][provider_upper]
        model_name = llm_config.get("model")
        temperature = llm_config.get("temperature", 0.7)
        max_tokens = llm_config.get("max_tokens", 2048)
        logging.info(f"LLM Config: provider={provider_upper}, model={model_name}, temp={temperature}, max_tokens={max_tokens}")
        
        # Get system prompt and rules from config based on role
        prompt_config = LLM_CONFIG.get("prompt_config", {})
        role_value = input_data.role.value
        role_config = prompt_config.get(role_value, prompt_config.get("default", {}))
        system_prompt = role_config.get("system_prompt", "You are a helpful assistant.")
        rules = role_config.get("Rules", [])
        logging.info(f"Using {len(rules)} rules for role: {role_value}")
        
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

        # Initialize LangChain's SQLChatMessageHistory
        chat_history = SQLChatMessageHistory(
            connection_string=POSTGRES_URL_CLEAN,
            session_id=session_id
        )
        
        # Get message history from LangChain
        messages = chat_history.messages
        
        # Apply hybrid history limits from config
        history_limits = LLM_CONFIG.get("history_limits", {})
        max_history_messages = history_limits.get("max_history_messages", 20)
        max_history_tokens = history_limits.get("max_history_tokens", 6000)
        max_history_age_days = history_limits.get("max_history_age_days", 7)
        
        # Filter messages based on hybrid limits
        filtered_messages = apply_history_limits(
            messages, 
            max_history_messages, 
            max_history_tokens, 
            max_history_age_days
        )
        
        # Initialize LLM with config settings using factory function
        llm = get_llm_handler(
            provider=provider_upper,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create prompt template with system prompt and message history
        if filtered_messages:
            # Create prompt with conversation history
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            
            # Create chain with history
            chain = prompt | llm
            response = chain.invoke({
                "history": filtered_messages,
                "input": input_data.prompt
            })
        else:
            # First message in conversation
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])
            
            chain = prompt | llm
            response = chain.invoke({"input": input_data.prompt})
        
        # Save messages to history in background (non-blocking)
        background_tasks.add_task(save_message_history, session_id, input_data.prompt, response)

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