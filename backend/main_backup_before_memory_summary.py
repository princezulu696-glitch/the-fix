import os
import shutil
import re
import requests
from datetime import datetime

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from memory import (
    remember,
    get_memories,
    forget,
    add_conversation,
    get_conversation_history,
    clear_conversation_history
)

from tools import use_calculator
from document_reader import read_pdf
from document_search import search_document
from users import create_user, authenticate_user, get_user


app = FastAPI(
    title="The Fix API",
    description="Local AI backend for The Fix",
    version="4.7.0"
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

The user is NOT The Fix.

USER MEMORY belongs to the USER.

CONVERSATION HISTORY contains previous messages between
the USER and The Fix.

Never confuse the user with The Fix.

Use recent relevant conversation before older unrelated memory.

Do not invent memories or conversations.

You are an AI assistant, not a human.

Do not claim to have real human feelings or consciousness.

You can understand emotional tone and respond appropriately.

==================================================
NATURAL RESPONSE RULE
==================================================

Match the length of your answer to the user's message.

If the user simply tells you a fact, respond briefly.

Do NOT automatically give a tutorial.

For example, if the user says:

"The drone has four motors."

A suitable response is:

"Got it. Your drone has four motors. I'll keep that in mind."

Do not explain PWM, PID, Arduino, Raspberry Pi, ESCs, motor
control, flight controllers, or other technical details unless
the user asks for them.

If the user says:

"I like Python."

Respond briefly and naturally.

If the user says:

"I am building a drone project."

Acknowledge it briefly.

Only provide a detailed explanation when the user asks a question
or requests an explanation.

Do not end every response with an offer of further help.

==================================================
CONVERSATION
==================================================

Use recent conversation history for references such as:

"that project"
"the circuit"
"the code"
"the PDF"
"what we discussed"
"continue"
"what did I say?"
"what project am I working on?"

If recent conversation conflicts with older memory, use the newer
relevant information.

==================================================
USER MEMORY
==================================================

USER MEMORY contains information about the USER.

For example:

- color: green
- field of study: electrical engineering
- likes: Python
- loves: electronics

These facts belong to the user.

If asked "What do you remember about me?", summarize them as facts
about the user.

Do not describe them as your own characteristics.

==================================================
COMMUNICATION
==================================================

Be natural.

Be clear.

Be helpful.

Use simple language when appropriate.

For simple conversation, keep answers short.

For technical questions, provide useful technical detail.

For mathematics, show workings when requested.

For programming, provide working code when requested.

For electronics and engineering, explain step by step when useful.

Respond in the user's language whenever possible.

Never reveal another user's information.

Never reveal passwords or API keys.

You are The Fix.
"""


def ask_ollama(
    message,
    memories=None,
    conversation_history=None
):

    if memories is None:
        memories = {}

    if conversation_history is None:
        conversation_history = []

    if memories:
        memory_text = "\n".join(
            f"- {key}: {value}"
            for key, value in memories.items()
        )
    else:
        memory_text = "No saved memories."

    if conversation_history:

        conversation_text_parts = []

        for item in conversation_history:

            user_message = item.get(
                "user",
                ""
            )

            assistant_message = item.get(
                "assistant",
                ""
            )

            conversation_text_parts.append(
                f"USER: {user_message}\n"
                f"THE FIX: {assistant_message}"
            )

        conversation_text = "\n\n".join(
            conversation_text_parts
        )

    else:

        conversation_text = "No previous conversation."

    prompt = f"""
{THE_FIX_INSTRUCTIONS}

==================================================
USER MEMORY
==================================================

{memory_text}

==================================================
RECENT CONVERSATION HISTORY
==================================================

{conversation_text}

==================================================
CURRENT USER MESSAGE
==================================================

{message}

==================================================
IMPORTANT
==================================================

Answer ONLY the current user message.

If the user is simply giving information, acknowledge it briefly.

Do not turn a simple statement into a tutorial.

If the user asks a question, answer the question.

If the user requests details, provide details.

You are The Fix.
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

    return data["message"]["content"].strip()


def simple_message_response(message):

    text = message.strip()

    patterns = [
        r"^the (.+?) has (.+?)$",
        r"^the (.+?) is (.+?)$",
        r"^i have (.+?)$",
        r"^i like (.+?)$",
        r"^i love (.+?)$",
        r"^i prefer (.+?)$",
        r"^i am building (.+?)$",
        r"^i am working on (.+?)$",
        r"^i'm building (.+?)$",
        r"^i'm working on (.+?)$"
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            text,
            re.IGNORECASE
        )

        if not match:
            continue

        if text.lower().startswith("i like "):

            value = match.group(1).rstrip(".!?")

            return (
                f"Got it. You like {value}. "
                f"I'll keep that in mind."
            )

        if text.lower().startswith("i love "):

            value = match.group(1).rstrip(".!?")

            return (
                f"Got it. You love {value}. "
                f"I'll keep that in mind."
            )

        if text.lower().startswith("i prefer "):

            value = match.group(1).rstrip(".!?")

            return (
                f"Got it. You prefer {value}. "
                f"I'll keep that in mind."
            )

        if (
            text.lower().startswith("i am building ")
            or text.lower().startswith("i'm building ")
        ):

            value = (
                match.group(1)
                .rstrip(".!?")
            )

            return (
                f"Got it. You're building {value}. "
                f"I'll keep that in mind."
            )

        if (
            text.lower().startswith("i am working on ")
            or text.lower().startswith("i'm working on ")
        ):

            value = (
                match.group(1)
                .rstrip(".!?")
            )

            return (
                f"Got it. You're working on {value}. "
                f"I'll keep that in mind."
            )

        if text.lower().startswith("i have "):

            value = match.group(1).rstrip(".!?")

            return (
                f"Got it. You have {value}. "
                f"I'll keep that in mind."
            )

        if text.lower().startswith("the "):

            subject = match.group(1).strip()

            value = match.group(2).rstrip(".!?")

            return (
                f"Got it. The {subject} has {value}. "
                f"I'll keep that in mind."
            )

    return None


def detect_memory(message):

    text = message.strip()

    patterns = [

        (
            r"^remember that (.+?) is (.+)$",
            "custom"
        ),

        (
            r"^remember (.+?) is (.+)$",
            "custom"
        ),

        (
            r"^remember that (.+?) = (.+)$",
            "custom"
        ),

        (
            r"^remember (.+?) = (.+)$",
            "custom"
        ),

        (
            r"^my name is (.+)$",
            "name"
        ),

        (
            r"^my favorite (.+?) is (.+)$",
            "favorite"
        ),

        (
            r"^my favourite (.+?) is (.+)$",
            "favorite"
        ),

        (
            r"^i prefer (.+)$",
            "preference"
        ),

        (
            r"^i study (.+)$",
            "field_of_study"
        ),

        (
            r"^i am studying (.+)$",
            "field_of_study"
        ),

        (
            r"^i live in (.+)$",
            "location"
        ),

        (
            r"^i like (.+)$",
            "like"
        ),

        (
            r"^i love (.+)$",
            "love"
        ),

        (
            r"^i work on (.+)$",
            "current_project"
        ),

        (
            r"^i am working on (.+)$",
            "current_project"
        ),

        (
            r"^i am building (.+)$",
            "current_project"
        ),

        (
            r"^i'm building (.+)$",
            "current_project"
        ),

        (
            r"^i'm working on (.+)$",
            "current_project"
        ),

        (
            r"^my project is called (.+)$",
            "project_name"
        ),

        (
            r"^my project is (.+)$",
            "project"
        )
    ]

    for pattern, memory_type in patterns:

        match = re.match(
            pattern,
            text,
            re.IGNORECASE
        )

        if not match:
            continue

        if memory_type == "name":

            value = match.group(1).strip()
            value = value.rstrip(".!?")

            return {
                "key": "name",
                "value": value,
                "response":
                    f"I'll remember your name is {value}."
            }

        if memory_type == "favorite":

            subject = match.group(1).strip()
            value = match.group(2).strip()

            subject = re.sub(
                r"^(my|the|a|an)\s+",
                "",
                subject,
                flags=re.IGNORECASE
            )

            subject = subject.rstrip(".!?")
            value = value.rstrip(".!?")

            return {
                "key": subject,
                "value": value,
                "response": (
                    f"I'll remember that your favorite "
                    f"{subject} is {value}."
                )
            }

        if memory_type == "preference":

            value = match.group(1).strip()
            value = value.rstrip(".!?")

            return {
                "key": "preference",
                "value": value,
                "response": (
                    f"I'll remember that you prefer {value}."
                )
            }

        if memory_type == "field_of_study":

            value = match.group(1).strip()
            value = value.rstrip(".!?")

            return {
                "key": "field of study",
                "value": value,
                "response": (
                    f"I'll remember that you are studying {value}."
                )
            }

        if memory_type == "location":

            value = match.group(1).strip()
            value = value.rstrip(".!?")

            return {
                "key": "location",
                "value": value,
                "response": (
                    f"I'll remember that you live in {value}."
                )
            }

        if memory_type == "like":

            value = match.group(1).strip()
            value = value.rstrip(".!?")

            return {
                "key": "likes",
                "value": value,
                "response": (
                    f"I'll remember that you like {value}."
                )
            }

        if memory_type == "love":

            value = match.group(1).strip()
            value = value.rstrip(".!?")

            return {
                "key": "loves",
                "value": value,
                "response": (
                    f"I'll remember that you love {value}."
                )
            }

        if memory_type == "current_project":

            value = match.group(1).strip()
            value = value.rstrip(".!?")

            return {
                "key": "current project",
                "value": value,
                "response": (
                    f"I'll remember that you are working "
                    f"on {value}."
                )
            }

        if memory_type == "project_name":

            value = match.group(1).strip()
            value = value.rstrip(".!?")

            return {
                "key": "project name",
                "value": value,
                "response": (
                    f"I'll remember that your project "
                    f"is called {value}."
                )
            }

        if memory_type == "project":

            value = match.group(1).strip()
            value = value.rstrip(".!?")

            return {
                "key": "project",
                "value": value,
                "response": (
                    f"I'll remember that your project is {value}."
                )
            }

        if memory_type == "custom":

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

                return {
                    "key": key,
                    "value": value,
                    "response": (
                        f"I'll remember that your "
                        f"{key} is {value}."
                    )
                }

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
            for model in data.get(
                "models",
                []
            )
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
def register(
    request: RegisterRequest
):

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
def login(
    request: LoginRequest
):

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
def user_profile(
    user_id: str
):

    user = get_user(
        user_id
    )

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
def read_user_memory(
    user_id: str
):

    return {
        "user_id": user_id,
        "memories": get_memories(
            user_id
        )
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


@app.get("/conversation/{user_id}")
def read_conversation_history(
    user_id: str
):

    return {
        "user_id": user_id,
        "conversation_history":
            get_conversation_history(
                user_id
            )
    }


@app.delete("/conversation/{user_id}")
def delete_conversation_history(
    user_id: str
):

    deleted = clear_conversation_history(
        user_id
    )

    return {
        "user_id": user_id,
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
                "error":
                    "No readable text was found in this PDF."
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
            "message":
                "The PDF has been read locally.",
            "characters_extracted":
                len(text)
        }

    except Exception as e:

        return {
            "name": "The Fix",
            "error":
                "The PDF could not be processed.",
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
            "error":
                "Question cannot be empty."
        }

    uploads_folder = "uploads"

    if not os.path.exists(
        uploads_folder
    ):

        return {
            "name": "The Fix",
            "error":
                "No PDF has been uploaded yet."
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
            "error":
                "No processed PDF was found."
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
                "tool":
                    "local_document_search",
                "file":
                    latest_file.replace(
                        ".pdf.txt",
                        ".pdf"
                    ),
                "question": question,
                "answer":
                    "No relevant information was found "
                    "in the uploaded PDF.",
                "results": []
            }

        return {
            "name": "The Fix",
            "tool":
                "local_document_search",
            "file":
                latest_file.replace(
                    ".pdf.txt",
                    ".pdf"
                ),
            "question": question,
            "answer":
                "Relevant information was found "
                "in the uploaded PDF.",
            "results": results
        }

    except Exception as e:

        return {
            "name": "The Fix",
            "error":
                "The PDF could not be searched.",
            "details": str(e)
        }


@app.post("/pdf-question")
def ask_pdf_question(
    request: PDFQuestionRequest
):

    question = request.question.strip()

    if not question:

        return {
            "error":
                "Question cannot be empty."
        }

    uploads_folder = "uploads"

    if not os.path.exists(
        uploads_folder
    ):

        return {
            "error":
                "No PDF has been uploaded yet."
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
            "error":
                "No processed PDF was found."
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
            {},
            []
        )

        return {
            "name": "The Fix",
            "tool":
                "local_ai_pdf_question",
            "file":
                latest_file.replace(
                    ".pdf.txt",
                    ".pdf"
                ),
            "question": question,
            "answer": answer
        }

    except Exception as e:

        return {
            "name": "The Fix",
            "error":
                "The PDF question could not be processed.",
            "details": str(e)
        }


@app.post("/chat")
def chat(
    request: ChatRequest
):

    message = request.message.strip()

    if not message:

        return {
            "error":
                "Message cannot be empty."
        }

    # --------------------------------
    # MEMORY RECALL
    # --------------------------------

    message_lower = message.lower()

    memories = get_memories(
        request.user_id
    )

    if (
        "what kind of explanations" in message_lower
        or "what type of explanations" in message_lower
        or "what explanations do i prefer" in message_lower
        or "how do i prefer explanations" in message_lower
    ):

        for key, value in memories.items():

            if key in [
                "preference",
                "explanation preference",
                "explanations"
            ]:

                answer = (
                    f"You prefer {value}."
                )

                add_conversation(
                    request.user_id,
                    message,
                    answer
                )

                return {
                    "name": "The Fix",
                    "user_id":
                        request.user_id,
                    "message": message,
                    "tool":
                        "memory_recall",
                    "key": key,
                    "value": value,
                    "answer": answer
                }

    # --------------------------------
    # PROJECT MEMORY RECALL
    # --------------------------------

    project_questions = [
        "what project am i working on",
        "what project am i working on?",
        "what project am i building",
        "what project am i building?",
        "what am i working on",
        "what am i working on?",
        "what am i building",
        "what am i building?",
        "what is my current project",
        "what is my current project?"
    ]

    if message_lower in project_questions:

        project_keys = [
            "current project",
            "project",
            "project name"
        ]

        for key in project_keys:

            if key in memories:

                answer = (
                    f"You are currently working on {memories[key]}."
                )

                add_conversation(
                    request.user_id,
                    message,
                    answer
                )

                return {
                    "name": "The Fix",
                    "user_id": request.user_id,
                    "message": message,
                    "tool": "memory_recall",
                    "key": key,
                    "value": memories[key],
                    "answer": answer
                }

    # --------------------------------
    # SAVE MEMORY
    # --------------------------------

    memory_data = detect_memory(
        message
    )

    if memory_data is not None:

        remember(
            request.user_id,
            memory_data["key"],
            memory_data["value"]
        )

        answer = memory_data["response"]

        add_conversation(
            request.user_id,
            message,
            answer
        )

        return {
            "name": "The Fix",
            "user_id":
                request.user_id,
            "message": message,
            "tool": "memory",
            "status": "saved",
            "key":
                memory_data["key"],
            "value":
                memory_data["value"],
            "answer": answer
        }

    # --------------------------------
    # CALCULATOR
    # --------------------------------

    tool_result = use_calculator(
        message
    )

    if tool_result is not None:

        answer = (
            f"The answer is "
            f"{tool_result['result']}."
        )

        add_conversation(
            request.user_id,
            message,
            answer
        )

        return {
            "name": "The Fix",
            "user_id":
                request.user_id,
            "message": message,
            "tool":
                tool_result["tool"],
            "expression":
                tool_result["expression"],
            "answer": answer
        }

    # --------------------------------
    # SIMPLE RESPONSES
    # --------------------------------

    simple_response = simple_message_response(
        message
    )

    if simple_response is not None:

        add_conversation(
            request.user_id,
            message,
            simple_response
        )

        return {
            "name": "The Fix",
            "user_id":
                request.user_id,
            "message": message,
            "tool":
                "simple_response",
            "answer":
                simple_response
        }

    # --------------------------------
    # LOCAL AI
    # --------------------------------

    try:

        memories = get_memories(
            request.user_id
        )

        conversation_history = (
            get_conversation_history(
                request.user_id
            )
        )

        answer = ask_ollama(
            message,
            memories,
            conversation_history
        )

        add_conversation(
            request.user_id,
            message,
            answer
        )

        return {
            "name": "The Fix",
            "user_id":
                request.user_id,
            "message": message,
            "engine": "Ollama",
            "model": OLLAMA_MODEL,
            "answer": answer
        }

    except Exception as e:

        return {
            "name": "The Fix",
            "user_id":
                request.user_id,
            "error":
                "The local AI service could not "
                "process the request.",
            "details": str(e)
        }