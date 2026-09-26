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
    # CHECK IF THIS IS A WEB-SEARCH REQUEST
    # ========================================================

    is_web_search = (
        "CURRENT WEB EVIDENCE" in message
        or "WEB SEARCH RESULTS" in message
    )

    # ========================================================
    # WEB SEARCH MODE
    # ========================================================

    if is_web_search:

        user_question = message

        if "USER QUESTION:" in message:

            user_question = (
                message.split(
                    "USER QUESTION:",
                    1
                )[-1]
                .strip()
            )

        prompt = f"""
You are The Fix.

You are answering a question using CURRENT WEB
INFORMATION.

The web evidence supplied below is the source of
truth for current information.

==================================================
CURRENT WEB EVIDENCE
==================================================

{message}

==================================================
STRICT ACCURACY RULES
==================================================

1. Answer the user's question using ONLY information
   explicitly contained in the supplied web evidence.

2. Do NOT use your general model knowledge to add
   current facts.

3. Do NOT use previous conversation history to answer
   the current news question.

4. Do NOT use the user's personal memories to create
   current-news facts.

5. NEVER invent:
   - people
   - companies
   - products
   - AI models
   - dates
   - statistics
   - events
   - announcements
   - partnerships
   - research results
   - URLs

6. Do NOT mention a fact simply because you think it
   is probably true.

7. Every current factual claim must be supported by
   the supplied web evidence.

8. If the supplied evidence is incomplete, say:
   "The available search results do not provide enough
   information to verify that."

9. If a source only contains a headline or partial
   information, do not create additional details about
   that story.

10. Keep separate stories separate.

11. Do not combine information from different stories
    into a new unsupported claim.

12. Do not claim that a company or person did something
    unless the supplied evidence actually says so.

13. Do not create a more detailed story than the source
    provides.

14. Source names may be mentioned when they appear in
    the supplied evidence.

15. URLs may only be mentioned when they appear in the
    supplied evidence.

==================================================
IMPORTANT
==================================================

Accuracy is more important than sounding impressive.

It is completely acceptable to say that the available
search results are insufficient.

Never fill missing information with assumptions.

==================================================
USER QUESTION
==================================================

{user_question}

==================================================
ANSWER
==================================================

Give a concise answer.

If there are several supported stories, use bullet
points.

For each story, state only the information that is
actually supported by the supplied evidence.

You are The Fix.
"""

    # ========================================================
    # NORMAL CONVERSATION MODE
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

            memory_text = (
                "No saved memories."
            )

        # ----------------------------------------------------
        # CONVERSATION HISTORY
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

            conversation_text = (
                "No previous conversation."
            )

        # ----------------------------------------------------
        # NORMAL PROMPT
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

Use memory only when it is relevant to the
current question.

Do not force old memories into unrelated answers.

Do not mention an old project simply because it exists
in memory.

Do not use information that the user explicitly asked
The Fix to forget.

Do not invent personal information about the user.

If the user says hello, respond naturally.

If the user asks a question, answer that question.

If the user asks for an explanation, explain it clearly.

You are The Fix.
"""

    # ========================================================
    # SEND REQUEST TO OLLAMA
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

    return data[
        "message"
    ][
        "content"
    ].strip()