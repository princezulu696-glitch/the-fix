import os
import shutil
import re
import json
import requests
from datetime import datetime

from web_search import web_search

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


# ============================================================
# THE FIX API
# ============================================================

app = FastAPI(
    title="The Fix API",
    description="Local AI backend for The Fix",
    version="4.9.0"
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


# ============================================================
# AI SETTINGS
# ============================================================

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen3:1.7b"


# ============================================================
# FORGOTTEN INFORMATION STORAGE
# ============================================================

FORGOTTEN_FILE = "forgotten_memory.json"


def load_forgotten_memory():
    """
    Load information that the user explicitly asked The Fix
    to forget.

    This is separate from normal memory because conversation
    history may still contain information that was forgotten.
    """

    if not os.path.exists(FORGOTTEN_FILE):
        return {}

    try:

        with open(
            FORGOTTEN_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(data, dict):
            return data

        return {}

    except Exception:

        return {}


def save_forgotten_memory(data):
    """
    Save forgotten information locally.
    """

    with open(
        FORGOTTEN_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


def add_forgotten_information(
    user_id,
    key,
    value=None
):
    """
    Record information that the user explicitly asked
    The Fix to forget.
    """

    forgotten = load_forgotten_memory()

    if user_id not in forgotten:
        forgotten[user_id] = []

    record = {
        "key": str(key).strip(),
        "value": (
            str(value).strip()
            if value is not None
            else None
        ),
        "forgotten_at": datetime.now().isoformat()
    }

    forgotten[user_id].append(record)

    save_forgotten_memory(
        forgotten
    )


def get_forgotten_information(user_id):
    """
    Return forgotten information for one user.
    """

    forgotten = load_forgotten_memory()

    records = forgotten.get(
        user_id,
        []
    )

    if isinstance(records, list):
        return records

    return []


def normalize_for_comparison(text):
    """
    Normalize text so forgotten information can be compared
    against old conversation history.
    """

    if text is None:
        return ""

    text = str(text).lower()

    text = text.replace(
        "’",
        "'"
    )

    text = re.sub(
        r"[^a-z0-9\s]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def text_contains_term(
    text,
    term
):
    """
    Check whether a normalized term appears inside text.
    """

    text_normalized = normalize_for_comparison(
        text
    )

    term_normalized = normalize_for_comparison(
        term
    )

    if not text_normalized or not term_normalized:
        return False

    return term_normalized in text_normalized


def conversation_entry_is_forgotten(
    item,
    forgotten_records
):
    """
    Determine whether a conversation entry contains information
    that the user explicitly asked The Fix to forget.

    Example:

    Forgotten:
        key = drone project
        value = Python

    Old conversation:
        My drone project uses Python.

    That conversation entry will not be sent to Ollama.
    """

    if not forgotten_records:
        return False

    user_message = item.get(
        "user",
        ""
    )

    assistant_message = item.get(
        "assistant",
        ""
    )

    combined_text = (
        f"{user_message}\n"
        f"{assistant_message}"
    )

    for record in forgotten_records:

        key = record.get(
            "key",
            ""
        )

        value = record.get(
            "value"
        )

        # ----------------------------------------------------
        # If both a key and value were forgotten,
        # require both to appear when possible.
        # ----------------------------------------------------

        if key and value:

            key_found = text_contains_term(
                combined_text,
                key
            )

            value_found = text_contains_term(
                combined_text,
                value
            )

            if key_found and value_found:
                return True

            # Also check common natural-language combinations.
            key_without_my = re.sub(
                r"^my\s+",
                "",
                key,
                flags=re.IGNORECASE
            )

            combined_patterns = [
                f"{key_without_my} uses {value}",
                f"{key_without_my} use {value}",
                f"{key_without_my} is {value}",
                f"{key_without_my} = {value}"
            ]

            for pattern in combined_patterns:

                if text_contains_term(
                    combined_text,
                    pattern
                ):
                    return True

        # ----------------------------------------------------
        # If only a value was supplied, filter conversations
        # containing that value.
        # ----------------------------------------------------

        elif value:

            if text_contains_term(
                combined_text,
                value
            ):
                return True

        # ----------------------------------------------------
        # If only a key was supplied, filter conversations
        # containing that key.
        # ----------------------------------------------------

        elif key:

            if text_contains_term(
                combined_text,
                key
            ):
                return True

    return False


def filter_conversation_history(
    user_id,
    conversation_history
):
    """
    Remove forgotten information from the context sent to Ollama.

    IMPORTANT:
    This does NOT destroy the user's entire conversation history.

    It only prevents explicitly forgotten information from being
    used as active AI context.
    """

    forgotten_records = get_forgotten_information(
        user_id
    )

    if not forgotten_records:
        return conversation_history

    filtered_history = []

    for item in conversation_history:

        if conversation_entry_is_forgotten(
            item,
            forgotten_records
        ):
            continue

        filtered_history.append(
            item
        )

    return filtered_history


# ============================================================
# REQUEST MODELS
# ============================================================

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


# ============================================================
# THE FIX SYSTEM INSTRUCTIONS
# ============================================================

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

If the user says:

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

Do not use information that has been explicitly marked as
FORGOTTEN.

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
FORGOTTEN INFORMATION
==================================================

If the user explicitly says they forgot, deleted, removed, or want
information forgotten:

1. Do not treat that information as active memory.
2. Do not retrieve it from conversation history.
3. Do not claim to remember it.
4. If asked about it later, say that the information is no longer
   available as an active memory.
5. Never recreate forgotten information from old conversation
   context.

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


# ============================================================
# OLLAMA
# ============================================================

def ask_ollama(
    message,
    memories=None,
    conversation_history=None
):

    if memories is None:
        memories = {}

    if conversation_history is None:
        conversation_history = []

    # ========================================================
    # DETECT WEB SEARCH
    # ========================================================

    is_web_search = (
        "WEB SEARCH RESULTS" in message
    )

    # ========================================================
    # WEB SEARCH MODE
    # ========================================================

    if is_web_search:

        web_information = message

        # ----------------------------------------------------
        # STRICT CURRENT-INFORMATION RULES
        # ----------------------------------------------------

        prompt = f"""
You are The Fix, a helpful AI assistant.

The user is asking about CURRENT or RECENT information.

The information below was retrieved from a web search.

Your job is to summarize ONLY what is supported by
the supplied web search evidence.

==================================================
WEB SEARCH EVIDENCE
==================================================

{web_information}

==================================================
STRICT EVIDENCE RULES
==================================================

1. Use the supplied web evidence as the primary and
   only source for current facts.

2. Do NOT use old conversation history to answer
   the current news question.

3. Do NOT use the user's saved memories to create
   current news facts.

4. Do NOT invent information.

5. Do NOT guess missing facts.

6. Do NOT add companies, people, products, dates,
   events, statistics, or claims unless they are
   supported by the supplied evidence.

7. If the evidence does not provide enough information,
   clearly say that the available search results do not
   provide enough information.

8. If a source only gives a headline or partial
   information, do not pretend that you know the
   complete story.

9. Keep different stories separate.

10. Do not combine unrelated stories into one claim.

11. When mentioning a story, mention the source name
    when possible.

12. If the evidence contains URLs, you may mention that
    the user can open the source for more details.

13. Do not claim that a source said something unless
    that information actually appears in the supplied
    evidence.

14. Never fill missing information using your own
    assumptions.

==================================================
ANSWER STYLE
==================================================

Give the user a concise answer.

Start with the important current information.

Use bullet points when there are several stories.

For each story:

- State only what the evidence supports.
- Mention the source.
- Do not invent additional details.

If the search evidence is weak, say so.

==================================================
USER QUESTION
==================================================

{message.split("USER QUESTION:")[-1].strip()}

==================================================

Remember:

You are The Fix.

Accuracy is more important than sounding confident.

If you do not know something from the supplied evidence,
say that you do not have enough information.
"""

    # ========================================================
    # NORMAL MODE
    # ========================================================

    else:

        # ----------------------------------------------------
        # MEMORY
        # ----------------------------------------------------

        if memories:

            memory_text = "\n".join(
                f"- {key}: {value}"
                for key, value in memories.items()
            )

        else:

            memory_text = "No saved memories."

        # ----------------------------------------------------
        # CONVERSATION
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # NORMAL AI PROMPT
        # ----------------------------------------------------

        prompt = f"""
{THE_FIX_INSTRUCTIONS}

==================================================
USER MEMORY
==================================================

{memory_text}

==================================================
RECENT CONVERSATION
==================================================

{conversation_text}

==================================================
CURRENT USER MESSAGE
==================================================

{message}

==================================================
IMPORTANT
==================================================

Answer the current user message directly.

Use saved memory when it is relevant.

Do not use information that the user explicitly
asked The Fix to forget.

Do not invent personal information about the user.

If the user gives a simple statement, acknowledge it
briefly.

If the user asks a question, answer it clearly.

You are The Fix.
"""

    # ========================================================
    # SEND TO OLLAMA
    # ========================================================

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

            value = match.group(1).rstrip(".!?")

            return (
                f"Got it. You're building {value}. "
                f"I'll keep that in mind."
            )

        if (
            text.lower().startswith("i am working on ")
            or text.lower().startswith("i'm working on ")
        ):

            value = match.group(1).rstrip(".!?")

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


# ============================================================
# FORGET REQUEST DETECTION
# ============================================================

def detect_forget_request(message):

    text = message.strip()
    lower_text = text.lower()

    patterns = [
        "forget that ",
        "forget ",
        "delete that ",
        "delete ",
        "remove that ",
        "remove "
    ]

    for pattern in patterns:

        if not lower_text.startswith(pattern):
            continue

        content = text[len(pattern):].strip()

        if content.lower().startswith("my "):
            content = content[3:].strip()

        if content.lower().startswith("that "):
            content = content[5:].strip()

        clean_content = content.rstrip(".!?").strip()

        # ----------------------------------------------------
        # "Forget that my drone project uses Python"
        # ----------------------------------------------------

        if " uses " in clean_content.lower():

            parts = re.split(
                r"\s+uses\s+",
                clean_content,
                maxsplit=1,
                flags=re.IGNORECASE
            )

            if len(parts) == 2:

                subject = parts[0].strip()
                value = parts[1].strip()

                if subject.lower().startswith("my "):
                    subject = subject[3:].strip()

                return {
                    "key": subject,
                    "value": value
                }

        # ----------------------------------------------------
        # "Forget that my favorite color is blue"
        # ----------------------------------------------------

        if " is " in clean_content.lower():

            parts = re.split(
                r"\s+is\s+",
                clean_content,
                maxsplit=1,
                flags=re.IGNORECASE
            )

            if len(parts) == 2:

                subject = parts[0].strip()
                value = parts[1].strip()

                if subject.lower().startswith("my "):
                    subject = subject[3:].strip()

                return {
                    "key": subject,
                    "value": value
                }

        # ----------------------------------------------------
        # "Forget Python"
        # ----------------------------------------------------

        return {
            "key": clean_content,
            "value": None
        }

    return None


# ============================================================
# MEMORY DETECTION
# ============================================================

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


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "The Fix",
        "status": "online",
        "engine": "Ollama",
        "model": OLLAMA_MODEL,
        "message": "The Fix API is running."
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "the-fix",
        "engine": "Ollama",
        "model": OLLAMA_MODEL,
        "time": datetime.now().isoformat()
    }


# ============================================================
# MODEL STATUS
# ============================================================

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


# ============================================================
# REGISTER
# ============================================================

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


# ============================================================
# LOGIN
# ============================================================

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


# ============================================================
# USER PROFILE
# ============================================================

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


# ============================================================
# SAVE MEMORY
# ============================================================

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


# ============================================================
# READ MEMORY
# ============================================================

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


# ============================================================
# DELETE MEMORY
# ============================================================

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


# ============================================================
# READ CONVERSATION
# ============================================================

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


# ============================================================
# DELETE CONVERSATION
# ============================================================

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


# ============================================================
# PDF UPLOAD
# ============================================================

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


# ============================================================
# PDF SEARCH
# ============================================================

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


# ============================================================
# PDF QUESTION
# ============================================================

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


# ============================================================
# CHAT
# ============================================================

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

    message_lower = message.lower()

    # --------------------------------------------------------
    # LOAD CURRENT SAVED MEMORY
    # --------------------------------------------------------

    memories = get_memories(
        request.user_id
    )

    # ========================================================
    # FORGET MEMORY
    # ========================================================

    forget_data = detect_forget_request(
        message
    )

    if forget_data is not None:

        key = forget_data["key"]

        value = forget_data.get(
            "value"
        )

        deleted = False

        # ----------------------------------------------------
        # Try exact key.
        # ----------------------------------------------------

        if key in memories:

            deleted = forget(
                request.user_id,
                key
            )

        # ----------------------------------------------------
        # Common aliases.
        # ----------------------------------------------------

        if not deleted:

            aliases = {
                "drone project": [
                    "current project",
                    "project",
                    "project name"
                ],

                "project": [
                    "current project",
                    "project",
                    "project name"
                ],

                "python": [
                    "likes",
                    "favorite programming language",
                    "programming language"
                ],

                "favorite color": [
                    "favorite color",
                    "colour"
                ],

                "favourite color": [
                    "favorite color",
                    "colour"
                ],

                "favourite colour": [
                    "favorite color",
                    "colour"
                ]
            }

            possible_keys = aliases.get(
                key.lower(),
                []
            )

            for possible_key in possible_keys:

                if possible_key in memories:

                    deleted = forget(
                        request.user_id,
                        possible_key
                    )

                    if deleted:
                        break

        # ----------------------------------------------------
        # Search saved memory values.
        # ----------------------------------------------------

        if not deleted and value:

            requested_value_lower = (
                value.lower()
            )

            for saved_key, saved_value in memories.items():

                if (
                    str(saved_value).lower()
                    == requested_value_lower
                ):

                    deleted = forget(
                        request.user_id,
                        saved_key
                    )

                    if deleted:
                        break

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Record the information as FORGOTTEN even if it was
        # not found in saved memory.
        #
        # This is what prevents old conversation history from
        # bringing it back.
        # ----------------------------------------------------

        add_forgotten_information(
            request.user_id,
            key,
            value
        )

        if deleted:

            answer = (
                "Done. I have forgotten that information."
            )

        else:

            answer = (
                "Done. I will no longer use that information "
                "as active memory."
            )

        # ----------------------------------------------------
        # Store the forget request itself in conversation.
        # ----------------------------------------------------

        add_conversation(
            request.user_id,
            message,
            answer
        )

        return {
            "name": "The Fix",
            "user_id": request.user_id,
            "message": message,
            "tool": "memory_forget",
            "status":
                "deleted"
                if deleted
                else "forgotten",
            "key": key,
            "value": value,
            "answer": answer
        }
   
    # ========================================================
    # FORGOTTEN INFORMATION RECALL
    # ========================================================

    forgotten_records = get_forgotten_information(
        request.user_id
    )

    if forgotten_records:

        question_text = normalize_for_comparison(message)

        for record in forgotten_records:

            forgotten_key = normalize_for_comparison(
                record.get("key", "")
            )

            forgotten_value = normalize_for_comparison(
                record.get("value", "")
            )

            # Check questions that clearly refer to
            # forgotten information.

            forgotten_question_patterns = [
                "what programming language does my drone project use",
                "what programming language does my drone use",
                "which programming language does my drone project use",
                "which programming language does my drone use",
                "what language does my drone project use",
                "what language does my drone use",
                "what programming language is my drone project using",
                "what language is my drone project using"
            ]

            question_is_about_forgotten_information = any(
                pattern in question_text
                for pattern in forgotten_question_patterns
            )

            key_match = (
                forgotten_key
                and forgotten_key in question_text
            )

            value_match = (
                forgotten_value
                and forgotten_value in question_text
            )

            if (
                question_is_about_forgotten_information
                or key_match
                or value_match
            ):

                answer = (
                    "I don't currently have that information "
                    "because you asked me to forget it."
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
                    "tool": "forgotten_memory",
                    "status": "forgotten",
                    "answer": answer
                }



    # ========================================================
    # MEMORY RECALL
    # ========================================================

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

    # ========================================================
    # MEMORY SUMMARY
    # ========================================================

    memory_summary_questions = [
        "what do you remember about me",
        "what do you remember about me?",
        "what do you know about me",
        "what do you know about me?",
        "what information do you remember about me",
        "what information do you remember about me?"
    ]

    if message_lower in memory_summary_questions:

        from memory import get_memory_summary

        summary = get_memory_summary(
            request.user_id
        )

        sections = []

        if summary["profile"]:

            sections.append(
                "Profile: "
                + ", ".join(
                    f"{key}: {value}"
                    for key, value
                    in summary["profile"].items()
                )
            )

        if summary["preferences"]:

            sections.append(
                "Preferences: "
                + ", ".join(
                    f"{key}: {value}"
                    for key, value
                    in summary["preferences"].items()
                )
            )

        if summary["projects"]:

            sections.append(
                "Projects: "
                + ", ".join(
                    f"{key}: {value}"
                    for key, value
                    in summary["projects"].items()
                )
            )

        if summary["interests"]:

            sections.append(
                "Interests: "
                + ", ".join(
                    f"{key}: {value}"
                    for key, value
                    in summary["interests"].items()
                )
            )

        if sections:

            answer = (
                "Here is what I remember about you:\n\n"
                + "\n".join(sections)
            )

        else:

            answer = (
                "I do not have any saved information "
                "about you yet."
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
            "tool": "memory_summary",
            "memory": summary,
            "answer": answer
        }

    # ========================================================
    # PROJECT MEMORY RECALL
    # ========================================================

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
                    f"You are currently working on "
                    f"{memories[key]}."
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

    # ========================================================
    # SAVE MEMORY
    # ========================================================

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

    # ========================================================
    # CALCULATOR
    # ========================================================

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

    # ========================================================
    # SIMPLE RESPONSES
    # ========================================================

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



    # ========================================================
    # WEB SEARCH
    # ========================================================

    web_search_result = None

    web_search_triggers = [
        "latest",
        "today",
        "current",
        "right now",
        "recent",
        "news",
        "weather",
        "price",
        "prices",
        "stock",
        "stocks",
        "score",
        "scores",
        "this week",
        "this month",
        "what happened",
        "who is the current",
        "who is currently",
        "current information"
    ]

    message_lower = message.lower()

    should_search_web = any(
        trigger in message_lower
        for trigger in web_search_triggers
    )

    if should_search_web:

        try:

            web_search_result = web_search(
                message
            )

        except Exception as error:

            web_search_result = {
                "success": False,
                "query": message,
                "answer": "",
                "source": "",
                "url": "",
                "results": [],
                "error": str(error)
            }


    # ========================================================
    # LOCAL AI
    # ========================================================

    try:

        # ----------------------------------------------------
        # Reload memory so the latest state is used.
        # ----------------------------------------------------

        memories = get_memories(
            request.user_id
        )

        # ----------------------------------------------------
        # Get complete conversation history.
        # ----------------------------------------------------

        conversation_history = (
            get_conversation_history(
                request.user_id
            )
        )

        # ----------------------------------------------------
        # Remove forgotten information before sending
        # anything to Ollama.
        # ----------------------------------------------------

        conversation_history = (
            filter_conversation_history(
                request.user_id,
                conversation_history
            )
        )

        # ----------------------------------------------------
        # WEB SEARCH CONTEXT
        # ----------------------------------------------------

        message_for_ai = message

        if (
            web_search_result
            and web_search_result.get("success")
        ):

            web_results = (
                web_search_result.get(
                    "results",
                    []
                )
            )

            web_context_parts = []

            for result in web_results:

                title = result.get(
                    "title",
                    ""
                )

                text = result.get(
                    "text",
                    ""
                )

                url = result.get(
                    "url",
                    ""
                )

                if text:

                    web_context_parts.append(
                        f"Title: {title}\n"
                        f"Information: {text}\n"
                        f"Source: {url}"
                    )

            web_context = "\n\n".join(
                web_context_parts
            )

            if web_context:

                message_for_ai = (
                    "You are The Fix.\n\n"
                    "The user is asking about information "
                    "that may require current web information.\n\n"
                    "Use the web search results below to "
                    "answer the user's question.\n\n"
                    "Do not invent current facts.\n"
                    "If the search results do not contain "
                    "enough information, say that clearly.\n\n"
                    "WEB SEARCH RESULTS:\n"
                    f"{web_context}\n\n"
                    "USER QUESTION:\n"
                    f"{message}"
                )

        # ----------------------------------------------------
        # Ask the local Ollama model.
        # ----------------------------------------------------

        answer = ask_ollama(
            message_for_ai,
            memories,
            conversation_history
        )

        # ----------------------------------------------------
        # Save the conversation.
        # ----------------------------------------------------

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
            "web_search":
                bool(
                    web_search_result
                    and web_search_result.get(
                        "success"
                    )
                ),
            "answer": answer
        }

    except Exception as error:

        return {
            "name": "The Fix",
            "user_id":
                request.user_id,
            "message": message,
            "error": str(error),
            "answer":
                "I encountered an error while "
                "processing your request."
        }