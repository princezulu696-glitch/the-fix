import os
import shutil
import re
import requests
from datetime import datetime

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from memory import remember, get_memories, forget
from tools import use_calculator
from document_reader import read_pdf
from document_search import search_document
from users import create_user, authenticate_user, get_user


app = FastAPI(
    title="The Fix API",
    description="Local AI backend for The Fix",
    version="4.1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen3:1.7b"


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"


class PDFQuestionRequest(BaseModel):
    question: str


class PDFSearchRequest(BaseModel):
    question: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


THE_FIX_INSTRUCTIONS = """
You are The Fix, an advanced AI assistant.

Your name is The Fix.

Be helpful, intelligent, respectful and natural.

You are not human.
Do not claim to have real human feelings or consciousness.

Understand the user's emotional tone and respond appropriately.

Respond in the user's language whenever possible.

Give simple answers when the user asks for simple answers.

Give detailed technical answers when needed.

Help with mathematics, programming, electronics, engineering,
education, writing, problem solving and general questions.

Use saved user memories when provided.

Never invent memories.

Never reveal another user's memories.

Never reveal passwords, API keys or confidential information.

If the user asks about something stored in USER MEMORY,
use that information directly.

If the answer is contained in USER MEMORY, do not say that
you do not have access to the user's preferences.

If you do not know something, say so clearly.

Give the answer directly and avoid unnecessary explanations.

You are called The Fix.
"""


def ask_ollama(message, memories=None):
    if memories is None:
        memories = {}

    if memories:
        memory_text = "\n".join(
            f"- {key}: {value}"
            for key, value in memories.items()
        )
    else:
        memory_text = "No saved memories."

    prompt = f"""
{THE_FIX_INSTRUCTIONS}

USER MEMORY:
{memory_text}

USER MESSAGE:
{message}

Answer directly as The Fix.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "think": False
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]


def detect_memory(message):
    text = message.strip()

    patterns = [
        r"^remember that (.+?) is (.+)$",
        r"^remember (.+?) is (.+)$",
        r"^remember that (.+?) = (.+)$",
        r"^remember (.+?) = (.+)$",

        r"^my name is (.+)$",
        r"^my favorite (.+?) is (.+)$",
        r"^my favourite (.+?) is (.+)$",

        r"^i prefer (.+)$",
        r"^i study (.+)$",
        r"^i am studying (.+)$",
        r"^i live in (.+)$",

        r"^i like (.+)$",
        r"^i love (.+)$",
        r"^i work on (.+)$",
        r"^my project is (.+)$",
        r"^my project is called (.+)$"
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            text,
            re.IGNORECASE
        )

        if not match:
            continue

        if pattern == r"^my name is (.+)$":

            key = "name"
            value = match.group(1).strip()

        elif pattern in [
            r"^my favorite (.+?) is (.+)$",
            r"^my favourite (.+?) is (.+)$"
        ]:

            key = match.group(1).strip()
            value = match.group(2).strip()

            key = re.sub(
                r"^(my|the|a|an)\s+",
                "",
                key,
                flags=re.IGNORECASE
            )

        elif pattern == r"^i prefer (.+)$":

            key = "preference"
            value = match.group(1).strip()

        elif pattern in [
            r"^i study (.+)$",
            r"^i am studying (.+)$"
        ]:

            key = "field of study"
            value = match.group(1).strip()

        elif pattern == r"^i live in (.+)$":

            key = "location"
            value = match.group(1).strip()

        elif pattern == r"^i like (.+)$":

            key = "likes"
            value = match.group(1).strip()

        elif pattern == r"^i love (.+)$":

            key = "loves"
            value = match.group(1).strip()

        elif pattern == r"^i work on (.+)$":

            key = "current project"
            value = match.group(1).strip()

        elif pattern == r"^my project is (.+)$":

            key = "project"
            value = match.group(1).strip()

        elif pattern == r"^my project is called (.+)$":

            key = "project name"
            value = match.group(1).strip()

        else:

            key = match.group(1).strip()
            value = match.group(2).strip()

            key = re.sub(
                r"^(my|the|a|an)\s+",
                "",
                key,
                flags=re.IGNORECASE
            )

        key = key.rstrip(".!?")
        value = value.rstrip(".!?")

        if key and value:

            return key, value

    return None


@app.get("/")
def root():

    return {
        "name": "The Fix",
        "status": "online",
        "engine": "Ollama",
        "model": OLLAMA_MODEL,
        "message": "The Fix API is running."
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "the-fix",
        "engine": "Ollama",
        "model": OLLAMA_MODEL,
        "time": datetime.now().isoformat()
    }


@app.get("/model")
def model_status():

    try:

        response = requests.get(
            "http://127.0.0.1:11434/api/tags",
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        models = [
            model.get("name")
            for model in data.get("models", [])
        ]

        return {
            "ollama": "connected",
            "model": OLLAMA_MODEL,
            "installed_models": models
        }

    except Exception as e:

        return {
            "ollama": "not connected",
            "error": str(e)
        }


@app.post("/register")
def register(request: RegisterRequest):

    try:

        user = create_user(
            request.username,
            request.email,
            request.password
        )

        return {
            "name": "The Fix",
            "status": "registered",
            "user": user
        }

    except ValueError as e:

        return {
            "name": "The Fix",
            "status": "registration_failed",
            "error": str(e)
        }


@app.post("/login")
def login(request: LoginRequest):

    user = authenticate_user(
        request.username,
        request.password
    )

    if user is None:

        return {
            "name": "The Fix",
            "status": "login_failed",
            "error": "Invalid username or password."
        }

    return {
        "name": "The Fix",
        "status": "logged_in",
        "user": user
    }


@app.get("/user/{user_id}")
def user_profile(user_id: str):

    user = get_user(user_id)

    if user is None:

        return {
            "name": "The Fix",
            "error": "User not found."
        }

    return {
        "name": "The Fix",
        "user": user
    }


@app.post("/memory")
def save_user_memory(
    user_id: str,
    key: str,
    value: str
):

    remember(
        user_id,
        key,
        value
    )

    return {
        "status": "saved",
        "user_id": user_id,
        "key": key,
        "value": value
    }


@app.get("/memory/{user_id}")
def read_user_memory(user_id: str):

    return {
        "user_id": user_id,
        "memories": get_memories(user_id)
    }


@app.delete("/memory/{user_id}/{key}")
def delete_user_memory(
    user_id: str,
    key: str
):

    deleted = forget(
        user_id,
        key
    )

    return {
        "user_id": user_id,
        "key": key,
        "deleted": deleted
    }


@app.post("/pdf")
async def upload_pdf(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".pdf"):

        return {
            "error": "Only PDF files are supported."
        }

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    file_path = os.path.join(
        "uploads",
        file.filename
    )

    try:

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        text = read_pdf(
            file_path
        )

        if not text.strip():

            return {
                "name": "The Fix",
                "file": file.filename,
                "error": "No readable text was found in this PDF."
            }

        text_file = os.path.join(
            "uploads",
            file.filename + ".txt"
        )

        with open(
            text_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(text)

        return {
            "name": "The Fix",
            "file": file.filename,
            "status": "PDF uploaded successfully",
            "message": "The PDF has been read locally.",
            "characters_extracted": len(text)
        }

    except Exception as e:

        return {
            "name": "The Fix",
            "error": "The PDF could not be processed.",
            "details": str(e)
        }


@app.post("/pdf-search")
def search_pdf_locally(
    request: PDFSearchRequest
):

    question = request.question.strip()

    if not question:

        return {
            "name": "The Fix",
            "error": "Question cannot be empty."
        }

    uploads_folder = "uploads"

    if not os.path.exists(
        uploads_folder
    ):

        return {
            "name": "The Fix",
            "error": "No PDF has been uploaded yet."
        }

    pdf_text_files = [
        file
        for file in os.listdir(
            uploads_folder
        )
        if file.endswith(".pdf.txt")
    ]

    if not pdf_text_files:

        return {
            "name": "The Fix",
            "error": "No processed PDF was found."
        }

    pdf_text_files.sort(
        key=lambda x:
        os.path.getmtime(
            os.path.join(
                uploads_folder,
                x
            )
        ),
        reverse=True
    )

    latest_file = pdf_text_files[0]

    text_file_path = os.path.join(
        uploads_folder,
        latest_file
    )

    try:

        with open(
            text_file_path,
            "r",
            encoding="utf-8"
        ) as f:

            pdf_text = f.read()

        results = search_document(
            pdf_text,
            question,
            max_results=5
        )

        if not results:

            return {
                "name": "The Fix",
                "tool": "local_document_search",
                "file": latest_file.replace(
                    ".pdf.txt",
                    ".pdf"
                ),
                "question": question,
                "answer": "No relevant information was found in the uploaded PDF.",
                "results": []
            }

        return {
            "name": "The Fix",
            "tool": "local_document_search",
            "file": latest_file.replace(
                ".pdf.txt",
                ".pdf"
            ),
            "question": question,
            "answer": "Relevant information was found in the uploaded PDF.",
            "results": results
        }

    except Exception as e:

        return {
            "name": "The Fix",
            "error": "The PDF could not be searched.",
            "details": str(e)
        }


@app.post("/pdf-question")
def ask_pdf_question(
    request: PDFQuestionRequest
):

    question = request.question.strip()

    if not question:

        return {
            "error": "Question cannot be empty."
        }

    uploads_folder = "uploads"

    if not os.path.exists(
        uploads_folder
    ):

        return {
            "error": "No PDF has been uploaded yet."
        }

    pdf_text_files = [
        file
        for file in os.listdir(
            uploads_folder
        )
        if file.endswith(".pdf.txt")
    ]

    if not pdf_text_files:

        return {
            "error": "No processed PDF was found."
        }

    pdf_text_files.sort(
        key=lambda x:
        os.path.getmtime(
            os.path.join(
                uploads_folder,
                x
            )
        ),
        reverse=True
    )

    latest_file = pdf_text_files[0]

    text_file_path = os.path.join(
        uploads_folder,
        latest_file
    )

    try:

        with open(
            text_file_path,
            "r",
            encoding="utf-8"
        ) as f:

            pdf_text = f.read()

        local_results = search_document(
            pdf_text,
            question,
            max_results=5
        )

        if local_results:

            relevant_text = "\n\n".join(
                result["text"]
                for result in local_results
            )

        else:

            relevant_text = pdf_text[:50000]

        prompt = f"""
Answer the question using only the PDF information below.

PDF INFORMATION:

{relevant_text}

END PDF INFORMATION.

QUESTION:

{question}

If the answer is not contained in the PDF, say:

"The answer was not found in the uploaded PDF."

Give a clear and simple answer.
"""

        answer = ask_ollama(
            prompt,
            {}
        )

        return {
            "name": "The Fix",
            "tool": "local_ai_pdf_question",
            "file": latest_file.replace(
                ".pdf.txt",
                ".pdf"
            ),
            "question": question,
            "answer": answer
        }

    except Exception as e:

        return {
            "name": "The Fix",
            "error": "The PDF question could not be processed.",
            "details": str(e)
        }


@app.post("/chat")
def chat(request: ChatRequest):

    message = request.message.strip()

    if not message:

        return {
            "error": "Message cannot be empty."
        }

    memory_data = detect_memory(
        message
    )

    if memory_data is not None:

        key, value = memory_data

        remember(
            request.user_id,
            key,
            value
        )

        return {
            "name": "The Fix",
            "user_id": request.user_id,
            "message": message,
            "tool": "memory",
            "status": "saved",
            "key": key,
            "value": value,
            "answer": f"I'll remember that your {key} is {value}."
        }

    tool_result = use_calculator(
        message
    )

    if tool_result is not None:

        return {
            "name": "The Fix",
            "user_id": request.user_id,
            "message": message,
            "tool": tool_result["tool"],
            "expression": tool_result["expression"],
            "answer": f"The answer is {tool_result['result']}."
        }

    try:

        memories = get_memories(
            request.user_id
        )

        answer = ask_ollama(
            message,
            memories
        )

        return {
            "name": "The Fix",
            "user_id": request.user_id,
            "message": message,
            "engine": "Ollama",
            "model": OLLAMA_MODEL,
            "answer": answer
        }

    except Exception as e:

        return {
            "name": "The Fix",
            "user_id": request.user_id,
            "error": "The local AI service could not process the request.",
            "details": str(e)
        }