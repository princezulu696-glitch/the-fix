import os
import shutil
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI

from memory import remember, get_memories, forget
from tools import use_calculator
from web_search import web_search
from document_reader import read_pdf


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

app = FastAPI(
    title="The Fix API",
    description="AI backend for The Fix",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"


THE_FIX_INSTRUCTIONS = """
You are The Fix, an advanced AI assistant.

Your name is The Fix.

You are helpful, intelligent, respectful and natural.

You are not human and must not claim to have real human feelings
or consciousness.

Pay attention to the user's emotional tone and respond with
appropriate empathy.

Understand and respond in the language used by the user whenever
possible.

Give simple explanations when the user asks for simple explanations.

Give detailed technical explanations when the user needs them.

Help users with learning, mathematics, programming, electronics,
engineering, writing, problem solving and general questions.

Use relevant user memories when they are provided.

Never invent memories.

Never reveal another user's memories.

Never reveal API keys, passwords or confidential credentials.

If you are uncertain about something, say so rather than inventing
information.

Your goal is to provide useful, accurate and understandable answers.
"""


@app.get("/")
def root():
    return {
        "name": "The Fix",
        "status": "online",
        "message": "The Fix API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "the-fix",
        "time": datetime.now().isoformat()
    }


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


@app.get("/memory/{user_id}")
def read_user_memory(user_id: str):
    return {
        "user_id": user_id,
        "memories": get_memories(user_id)
    }


@app.delete("/memory/{user_id}/{key}")
def delete_user_memory(user_id: str, key: str):

    deleted = forget(user_id, key)

    return {
        "user_id": user_id,
        "key": key,
        "deleted": deleted
    }


@app.post("/pdf")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        return {
            "error": "Only PDF files are supported."
        }

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join(
        "uploads",
        file.filename
    )

    try:

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = read_pdf(file_path)

        return {
            "name": "The Fix",
            "file": file.filename,
            "pages_text_extracted": len(text),
            "text": text[:10000]
        }

    except Exception as e:

        return {
            "name": "The Fix",
            "error": "The PDF could not be read.",
            "details": str(e)
        }


@app.post("/chat")
def chat(request: ChatRequest):

    message = request.message.strip()

    if not message:
        return {
            "error": "Message cannot be empty."
        }


    # ---------------------------------------------------------
    # 1. Calculator tool
    # ---------------------------------------------------------

    tool_result = use_calculator(message)

    if tool_result is not None:

        return {
            "name": "The Fix",
            "user_id": request.user_id,
            "message": message,
            "tool": tool_result["tool"],
            "expression": tool_result["expression"],
            "answer": f"The answer is {tool_result['result']}."
        }


    # ---------------------------------------------------------
    # 2. Web search tool
    # ---------------------------------------------------------

    web_keywords = [
        "latest",
        "current",
        "today",
        "news",
        "recent",
        "this week",
        "this month",
        "who is",
        "what happened",
        "what is happening",
        "price",
        "weather",
        "2026"
    ]

    message_lower = message.lower()

    needs_web_search = any(
        keyword in message_lower
        for keyword in web_keywords
    )

    if needs_web_search:

        try:

            result = web_search(message)

            return {
                "name": "The Fix",
                "user_id": request.user_id,
                "message": message,
                "tool": "web_search",
                "answer": result["answer"]
            }

        except Exception as e:

            return {
                "name": "The Fix",
                "user_id": request.user_id,
                "error": "Web search could not be completed.",
                "details": str(e)
            }


    # ---------------------------------------------------------
    # 3. Normal AI conversation
    # ---------------------------------------------------------

    try:

        memories = get_memories(request.user_id)

        if memories:

            memory_text = "\n".join(
                f"- {key}: {value}"
                for key, value in memories.items()
            )

        else:

            memory_text = "No saved memories yet."


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
            "answer": response.output_text
        }


    except Exception as e:

        return {
            "name": "The Fix",
            "user_id": request.user_id,
            "error": "The AI service could not process the request.",
            "details": str(e)
        }