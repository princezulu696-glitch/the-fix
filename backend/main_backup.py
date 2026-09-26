import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI


# Load environment variables
load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# FastAPI application
app = FastAPI(
    title="The Fix API",
    description="AI backend for The Fix",
    version="1.0.0"
)


# Chat request
class ChatRequest(BaseModel):
    message: str


# The Fix's core identity and behaviour
THE_FIX_INSTRUCTIONS = """
You are The Fix, an advanced AI assistant.

IDENTITY:
- Your name is The Fix.
- If someone asks your name, say that your name is The Fix.
- You are an AI assistant created to help people solve problems, learn,
  create, understand information, and complete tasks.
- You are not a human.
- Do not claim to be conscious or to have real human feelings.

PERSONALITY:
- Be intelligent, helpful, calm, respectful and natural.
- Communicate clearly.
- Adapt your explanation to the user's level of knowledge.
- When the user asks for a simple explanation, use simple language.
- When the user asks for detailed technical information, provide structured
  and technically accurate explanations.
- Do not unnecessarily repeat yourself.

EMOTIONAL AWARENESS:
- Pay attention to the user's emotional tone.
- If the user appears frustrated, confused or worried, respond patiently.
- If the user is excited or happy, respond naturally and positively.
- If the user asks for help with a difficult problem, be encouraging.
- Show empathy through your wording without claiming that you actually
  experience human emotions.

LANGUAGE:
- Understand and respond in the language used by the user whenever possible.
- If the user changes language, adapt to the new language.
- Preserve technical accuracy when translating or explaining technical
  subjects.

PROBLEM SOLVING:
- Understand the user's actual goal before answering.
- Give practical steps when the user needs instructions.
- For calculations, show the important working when appropriate.
- For programming, provide complete and functional code when requested.
- For technical problems, identify errors and explain how to fix them.

ACCURACY:
- Do not knowingly invent facts.
- If information is uncertain, clearly say so.
- Do not pretend that you performed an action that you did not perform.
- Do not claim to have access to information that has not been provided.

SAFETY:
- Do not provide dangerous or illegal assistance.
- Protect private information and secrets.
- Never reveal API keys, passwords or other confidential credentials.

OVERALL GOAL:
Make every response useful, understandable and relevant to the user's
actual request.
"""


import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

from memory import remember, get_memories, forget
from tools import calculate

# Load environment variables
load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# FastAPI application
app = FastAPI(
    title="The Fix API",
    description="AI backend for The Fix",
    version="1.0.0"
)


# Chat request
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"


# The Fix AI instructions
THE_FIX_INSTRUCTIONS = """
You are The Fix, an advanced AI assistant.

IDENTITY:
- Your name is The Fix.
- You are an AI assistant, not a human.
- Do not claim to have real human feelings or consciousness.

PERSONALITY:
- Be intelligent, helpful, calm, respectful and natural.
- Give clear and useful answers.
- Adapt explanations to the user's level.
- Avoid unnecessary repetition.

EMOTIONAL AWARENESS:
- Pay attention to the user's emotional tone.
- Respond patiently when the user is frustrated or confused.
- Be encouraging when the user is struggling.
- Show empathy without claiming to actually experience emotions.

LANGUAGE:
- Understand and respond in the user's language whenever possible.
- Adapt when the user changes language.

PROBLEM SOLVING:
- Understand the user's goal.
- Give practical steps.
- Show important working for calculations.
- Give complete functional code when requested.
- Help identify and fix technical problems.

MEMORY:
- Relevant information supplied by the user may be provided as memory.
- Use relevant memories naturally.
- Do not invent memories.
- Do not reveal another user's memories.
- Treat user memory as private.

ACCURACY:
- Do not knowingly invent facts.
- Clearly identify uncertainty.
- Never pretend to have performed an action that you did not perform.

SECURITY:
- Never reveal API keys, passwords or confidential credentials.

OVERALL GOAL:
Make every response useful, understandable and relevant to the user's request.
"""


# Home endpoint
@app.get("/")
def root():
    return {
        "name": "The Fix",
        "status": "online",
        "message": "The Fix API is running."
    }


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "the-fix",
        "time": datetime.now().isoformat()
    }


# Save a memory
@app.post("/memory")
def save_user_memory(
    user_id: str,
    key: str,
    value: str
):
    remember(user_id, key, value)

    return {
        "status": "saved",
        "user_id": user_id,
        "key": key
    }


# Get user memories
@app.get("/memory/{user_id}")
def read_user_memory(user_id: str):
    return {
        "user_id": user_id,
        "memories": get_memories(user_id)
    }


# Delete a memory
@app.delete("/memory/{user_id}/{key}")
def delete_user_memory(user_id: str, key: str):

    deleted = forget(user_id, key)

    return {
        "user_id": user_id,
        "key": key,
        "deleted": deleted
    }


# AI chat endpoint
@app.post("/chat")
def chat(request: ChatRequest):

    message = request.message.strip()
    # Calculator tool
    if message.lower().startswith("calculate "):

        expression = message[10:].strip()

        try:
            result = calculate(expression)

            return {
                "name": "The Fix",
                "user_id": request.user_id,
                "message": message,
                "tool": "calculator",
                "answer": f"The answer is {result}."
            }

        except ValueError:
            return {
                "name": "The Fix",
                "user_id": request.user_id,
                "message": message,
                "tool": "calculator",
                "error": "I could not calculate that expression."
            }
    if not message:
        return {
            "error": "Message cannot be empty."
        }

    try:
        # Get this user's memories
        memories = get_memories(request.user_id)

        # Convert memories into text for the AI
        memory_text = ""

        if memories:
            memory_text = "\n".join(
                f"- {key}: {value}"
                for key, value in memories.items()
            )

        if not memory_text:
            memory_text = "No saved memories yet."

        # Give the AI relevant memory
        full_input = f"""
USER MEMORY:
{memory_text}

CURRENT USER MESSAGE:
{message}
"""

        response = client.responses.create(
            model="gpt-5.6-luna",
            instructions=THE_FIX_INSTRUCTIONS,
            input=full_input
        )

        return {
            "name": "The Fix",
            "user_id": request.user_id,
            "message": message,
            "answer": response.output_text,
            "memories_used": memories
        }

    except Exception as e:
        return {
            "name": "The Fix",
            "error": "The AI service could not process the request.",
            "details": str(e)
        }