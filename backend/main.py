import os
import time
import re
import requests
from pathlib import Path

from math_engine import solve_math
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from vision import router as vision_router

from users import (
    create_user,
    authenticate_user,
    get_user,
)

from memory import (
    remember,
    get_memories,
    get_memory_summary,
    forget,
    add_conversation,
    get_conversation_history,
    clear_conversation_history,
)

from tools import use_calculator
from web_search import web_search


# ============================================================
# OPTIONAL DOCUMENT SYSTEM
# ============================================================

try:
    from document_reader import read_document
except Exception:
    read_document = None

try:
    from document_search import search_document
except Exception:
    search_document = None


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# APPLICATION
# ============================================================

APP_NAME = "The Fix"
APP_VERSION = "11.3.0"

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"


# ============================================================
# MODELS
# ============================================================

FAST_MODEL = "qwen3:1.7b"
POWERFUL_MODEL = "qwen3:4b"
VISION_MODEL = "qwen3-vl:2b"
MORENA_MODEL = "hf.co/vamboai/morena-1.5b-instruct-gguf:Q4_K_M"

OLLAMA_MODEL = FAST_MODEL


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

FAST_KEEP_ALIVE = "2h"
MORENA_KEEP_ALIVE = "2h"
POWERFUL_KEEP_ALIVE = "30m"


# ============================================================
# MODEL CACHE
# ============================================================

_MODEL_CACHE = []
_MODEL_CACHE_TIME = 0
MODEL_CACHE_SECONDS = 60


# ============================================================
# FRONTEND
# ============================================================

FRONTEND_FOLDER = (
    Path(__file__).resolve().parent.parent / "frontend"
)

FRONTEND_FILE = FRONTEND_FOLDER / "index.html"


# ============================================================
# LANGUAGES
# ============================================================

LANGUAGES = {
    "en-ZA": "English (South Africa)",
    "zu-ZA": "isiZulu",
    "xh-ZA": "isiXhosa",
    "st-ZA": "Sesotho",
    "tn-ZA": "Setswana",
    "nso-ZA": "Sepedi",
    "af-ZA": "Afrikaans",
    "ss-ZA": "siSwati",
    "ts-ZA": "itsonga",
    "ve-ZA": "Tshivenda",
    "nr-ZA": "isiNdebele",

    "en-US": "English",
    "fr-FR": "French",
    "es-ES": "Spanish",
    "pt-PT": "Portuguese",
    "de-DE": "German",
    "it-IT": "Italian",
    "nl-NL": "Dutch",
    "ar-SA": "Arabic",
    "zh-CN": "Chinese",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "hi-IN": "Hindi",
    "ru-RU": "Russian",
}


# ============================================================
# NATIVE LANGUAGE NAMES
# ============================================================

LANGUAGE_NATIVE_NAMES = {
    "en-ZA": "English",
    "zu-ZA": "isiZulu",
    "xh-ZA": "isiXhosa",
    "st-ZA": "Sesotho",
    "tn-ZA": "Setswana",
    "nso-ZA": "Sepedi",
    "af-ZA": "Afrikaans",
    "ss-ZA": "siSwati",
    "ts-ZA": "itsonga",
    "ve-ZA": "Tshivenda",
    "nr-ZA": "isiNdebele",

    "en-US": "English",
    "fr-FR": "Français",
    "es-ES": "Español",
    "pt-PT": "Português",
    "de-DE": "Deutsch",
    "it-IT": "Italiano",
    "nl-NL": "Nederlands",
    "ar-SA": "العربية",
    "zh-CN": "中文",
    "ja-JP": "日本語",
    "ko-KR": "한국어",
    "hi-IN": "हिन्दी",
    "ru-RU": "Русский",
}


AFRICAN_LANGUAGES = {
    "zu-ZA",
    "xh-ZA",
    "st-ZA",
    "tn-ZA",
    "nso-ZA",
    "af-ZA",
    "ss-ZA",
    "ts-ZA",
    "ve-ZA",
    "nr-ZA",
}


MULTILINGUAL_LANGUAGES = AFRICAN_LANGUAGES | {
    "fr-FR",
    "es-ES",
    "pt-PT",
    "de-DE",
    "it-IT",
    "nl-NL",
    "ar-SA",
    "zh-CN",
    "ja-JP",
    "ko-KR",
    "hi-IN",
    "ru-RU",
}


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="The Fix - multilingual personal AI",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(vision_router)


# ============================================================
# REQUEST MODELS
# ============================================================

class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class MemoryRequest(BaseModel):
    user_id: str
    key: str
    value: str


class ForgetRequest(BaseModel):
    user_id: str
    key: str


class ChatRequest(BaseModel):
    user_id: str
    message: str
    language: str = "en-ZA"


class ConversationClearRequest(BaseModel):
    user_id: str


class DocumentQuestionRequest(BaseModel):
    user_id: str
    filename: str
    question: str
    language: str = "en-ZA"


class DocumentSearchRequest(BaseModel):
    filename: str
    query: str
    top_k: int = 5


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_string(value):
    if value is None:
        return ""

    return str(value).strip()


def get_language_code(language):
    language = safe_string(language)

    if language in LANGUAGES:
        return language

    return "en-ZA"


def get_language_name(language):
    code = get_language_code(language)

    return LANGUAGES.get(
        code,
        "English (South Africa)"
    )


def get_language_native_name(language):
    code = get_language_code(language)

    return LANGUAGE_NATIVE_NAMES.get(
        code,
        "English"
    )


def is_multilingual_language(language):
    return get_language_code(language) in MULTILINGUAL_LANGUAGES


def is_african_language(language):
    return get_language_code(language) in AFRICAN_LANGUAGES


# ============================================================
# LIGHTWEIGHT LANGUAGE DETECTION
# ============================================================

ENGLISH_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "hiya",
    "good morning",
    "good afternoon",
    "good evening",
    "morning",
    "afternoon",
    "evening",
}


ZULU_GREETINGS = {
    "sawubona",
    "sanibonani",
    "yebo",
    "unjani",
    "ninjani",
    "ngiyabingelela",
}


ENGLISH_MARKERS = {
    "the",
    "is",
    "are",
    "am",
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "can",
    "could",
    "would",
    "should",
    "please",
    "explain",
    "tell",
    "give",
    "help",
    "calculate",
    "solve",
    "show",
    "make",
    "create",
    "want",
    "need",
    "this",
    "that",
    "with",
    "from",
    "for",
    "and",
    "or",
    "about",
    "hello",
    "hi",
    "hey",
    "electricity",
    "computer",
    "python",
    "code",
}


ZULU_MARKERS = {
    "ngicela",
    "ngiyacela",
    "ukuthi",
    "uyini",
    "yini",
    "kanjani",
    "kungani",
    "kuphi",
    "nini",
    "ngubani",
    "ngiyabonga",
    "sawubona",
    "sanibonani",
    "unjani",
    "ninjani",
    "mina",
    "wena",
    "thina",
    "lokhu",
    "leyo",
    "lena",
    "lapha",
    "khona",
    "kakhulu",
    "ngifuna",
    "ngidinga",
    "ngisize",
    "ngichazele",
    "ngitshele",
    "futhi",
    "kodwa",
    "ngoba",
    "uma",
    "yebo",
    "cha",
    "ugesi",
    "amanzi",
    "isikole",
    "umsebenzi",
    "umbuzo",
    "impendulo",
}


def normalize_for_language_detection(text):
    text = safe_string(text).lower()

    text = re.sub(
        r"[^\w\s?'!-]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def detect_input_language(message, requested_language):

    requested = get_language_code(
        requested_language
    )

    text = normalize_for_language_detection(
        message
    )

    if not text:
        return requested

    if text in ENGLISH_GREETINGS:

        if requested == "en-US":
            return "en-US"

        return "en-ZA"

    if text in ZULU_GREETINGS:
        return "zu-ZA"

    words = set(
        text.split()
    )

    english_score = 0
    zulu_score = 0

    for word in words:

        if word in ENGLISH_MARKERS:
            english_score += 1

        if word in ZULU_MARKERS:
            zulu_score += 1

    if zulu_score >= 2 and zulu_score > english_score:
        return "zu-ZA"

    if english_score >= 2 and english_score > zulu_score:

        if requested == "en-US":
            return "en-US"

        return "en-ZA"

    distinctive_zulu = {
        "ngicela",
        "ngiyacela",
        "sawubona",
        "sanibonani",
        "ngiyabonga",
        "uyini",
        "kungani",
        "kanjani",
        "ngisize",
        "ngichazele",
        "ngitshele",
        "ugesi",
    }

    if any(
        word in words
        for word in distinctive_zulu
    ):
        return "zu-ZA"

    return requested


# ============================================================
# SIMPLE GREETING DETECTION
# ============================================================

def detect_greeting(message):

    text = normalize_for_language_detection(
        message
    )

    if text in ENGLISH_GREETINGS:
        return "en"

    if text in ZULU_GREETINGS:
        return "zu"

    return None


def greeting_response(language):

    language = get_language_code(
        language
    )

    if language == "zu-ZA":
        return "Sawubona! Ngingakusiza ngani?"

    return "Hello! How can I assist you today?"


# ============================================================
# OLLAMA MODEL LIST
# ============================================================

def get_available_models(force=False):

    global _MODEL_CACHE
    global _MODEL_CACHE_TIME

    now = time.time()

    if (
        not force
        and _MODEL_CACHE
        and now - _MODEL_CACHE_TIME < MODEL_CACHE_SECONDS
    ):
        return _MODEL_CACHE

    try:

        response = requests.get(
            OLLAMA_TAGS_URL,
            timeout=3
        )

        if response.status_code != 200:
            return _MODEL_CACHE

        data = response.json()

        models = [
            item.get("name")
            for item in data.get("models", [])
            if item.get("name")
        ]

        _MODEL_CACHE = models
        _MODEL_CACHE_TIME = now

        return models

    except Exception:

        return _MODEL_CACHE


def model_available(model_name):
    return model_name in get_available_models()


# ============================================================
# REQUEST COMPLEXITY
# ============================================================

def is_complex_request(message):

    text = safe_string(message).lower()

    complex_phrases = [
        "explain in detail",
        "detailed explanation",
        "step by step",
        "analyse",
        "analyze",
        "summarize",
        "summary",
        "compare",
        "calculate",
        "solve",
        "derive",
        "prove",
        "equation",
        "mathematics",
        "math",
        "engineering",
        "physics",
        "thermodynamics",
        "laplace",
        "differential equation",
        "program",
        "programming",
        "code",
        "debug",
        "python",
        "javascript",
        "matlab",
        "algorithm",
        "report",
        "essay",
        "design",
        "everything about",
        "why does",
        "how does",
        "explain how",
        "explain why",
    ]

    for phrase in complex_phrases:

        if phrase in text:
            return True

    if len(text) > 280:
        return True

    return False


# ============================================================
# MODEL ROUTER
# ============================================================

def choose_model(
    user_message,
    language="en-ZA",
    has_web=False,
    has_document=False
):

    language_code = get_language_code(
        language
    )

    if language_code == "zu-ZA":

        if model_available(MORENA_MODEL):
            return MORENA_MODEL, "morena-isizulu"

        if model_available(FAST_MODEL):
            return FAST_MODEL, "isizulu-fallback"

        return POWERFUL_MODEL, "isizulu-fallback"

    if has_document or has_web:

        if model_available(POWERFUL_MODEL):

            if is_multilingual_language(
                language_code
            ):
                return POWERFUL_MODEL, "multilingual-powerful"

            return POWERFUL_MODEL, "powerful"

        return FAST_MODEL, "fast"

    if is_complex_request(
        user_message
    ):

        if model_available(POWERFUL_MODEL):

            if is_multilingual_language(
                language_code
            ):
                return POWERFUL_MODEL, "multilingual-powerful"

            return POWERFUL_MODEL, "powerful"

        return FAST_MODEL, "fast"

    if is_multilingual_language(
        language_code
    ):
        return FAST_MODEL, "multilingual-fast"

    return FAST_MODEL, "fast"


# ============================================================
# MEMORY
# ============================================================

def get_memory_context(user_id):

    try:

        memories = get_memories(
            user_id
        )

        if not memories:
            return ""

        return str(
            memories
        )

    except Exception:

        return ""


def get_conversation_context(user_id):

    try:

        history = get_conversation_history(
            user_id
        )

        if not history:
            return ""

        if isinstance(
            history,
            list
        ):

            recent = history[-4:]

            lines = []

            for item in recent:

                if isinstance(
                    item,
                    dict
                ):

                    role = item.get(
                        "role",
                        ""
                    )

                    content = item.get(
                        "content",
                        item.get(
                            "message",
                            ""
                        )
                    )

                    if content:

                        content = str(
                            content
                        )[:1000]

                        lines.append(
                            f"{role}: {content}"
                        )

            return "\n".join(
                lines
            )

        return str(
            history
        )[-4000:]

    except Exception:

        return ""


# ============================================================
# AUTOMATIC MEMORY
# ============================================================

def detect_and_save_memory(
    user_id,
    message
):

    text = safe_string(
        message
    )

    patterns = [

        (
            r"\bmy name is\s+(.+)",
            "name"
        ),

        (
            r"\bcall me\s+(.+)",
            "name"
        ),

        (
            r"\bremember that my name is\s+(.+)",
            "name"
        ),

        (
            r"\bmy favorite color is\s+(.+)",
            "favorite_color"
        ),

        (
            r"\bmy favourite color is\s+(.+)",
            "favorite_color"
        ),

        (
            r"\bmy favorite subject is\s+(.+)",
            "favorite_subject"
        ),

        (
            r"\bmy favourite subject is\s+(.+)",
            "favorite_subject"
        ),

        (
            r"\bmy programming language is\s+(.+)",
            "programming_language"
        ),

        (
            r"\bi prefer\s+(.+)",
            "preference"
        ),

        (
            r"\bi am building\s+(.+)",
            "current_project"
        ),
    ]

    for pattern, key in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            value = match.group(
                1
            ).strip()

            value = re.sub(
                r"[.!?]+$",
                "",
                value
            ).strip()

            if value:

                try:

                    remember(
                        user_id,
                        key,
                        value
                    )

                    return True

                except Exception:

                    return False

    return False


# ============================================================
# WEB SEARCH
# ============================================================

def needs_web_search(message):

    text = safe_string(
        message
    ).lower()

    keywords = [
        "latest",
        "today",
        "current",
        "currently",
        "news",
        "recent",
        "this week",
        "this month",
        "price",
        "prices",
        "weather",
        "stock price",
        "exchange rate",
        "who is the current",
        "what happened today",
        "latest update",
    ]

    return any(
        keyword in text
        for keyword in keywords
    )


# ============================================================
# LANGUAGE INSTRUCTIONS
# ============================================================

def build_language_instructions(
    language_code,
    language_name,
    native_name
):

    return f"""
LANGUAGE RULE:

The user's detected language is the priority.

Detected language:
{language_name} ({language_code})

Native language name:
{native_name}

Answer in the same language as the user's message.

Do not unnecessarily switch to English.

Do not translate the user's question unless the user asks
for translation.

Do not repeat the user's question.

Give the actual answer.
"""


# ============================================================
# RESPONSE CLEANING / ANTI-REPETITION
# ============================================================

def clean_repeated_response(answer):

    answer = safe_string(
        answer
    )

    if not answer:
        return answer

    answer = re.sub(
        r"<think>.*?</think>",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE
    ).strip()

    prefixes = [
        "Thinking...",
        "Let me think...",
        "Let me analyze...",
        "Let me analyse...",
        "First, I need to understand...",
    ]

    changed = True

    while changed:

        changed = False

        for prefix in prefixes:

            if answer.lower().startswith(
                prefix.lower()
            ):

                answer = answer[
                    len(prefix):
                ].strip()

                changed = True

    lines = answer.splitlines()

    cleaned_lines = []
    seen_lines = set()

    for line in lines:

        clean_line = line.strip()

        if not clean_line:
            continue

        normalized = re.sub(
            r"\s+",
            " ",
            clean_line.lower()
        )

        if normalized in seen_lines:
            continue

        seen_lines.add(
            normalized
        )

        cleaned_lines.append(
            clean_line
        )

    answer = "\n".join(
        cleaned_lines
    ).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer
    )

    final_sentences = []
    seen_sentences = set()

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        normalized = re.sub(
            r"\s+",
            " ",
            sentence.lower()
        ).strip()

        if normalized in seen_sentences:
            continue

        seen_sentences.add(
            normalized
        )

        final_sentences.append(
            sentence
        )

    answer = " ".join(
        final_sentences
    ).strip()

    words = answer.split()

    if len(words) >= 30:

        max_block = min(
            20,
            len(words) // 2
        )

        for size in range(
            5,
            max_block + 1
        ):

            first = [
                re.sub(
                    r"[^\w]",
                    "",
                    word.lower()
                )
                for word in words[:size]
            ]

            second = [
                re.sub(
                    r"[^\w]",
                    "",
                    word.lower()
                )
                for word in words[size:size * 2]
            ]

            if (
                first
                and first == second
            ):

                answer = " ".join(
                    words[:size]
                ).strip()

                break

    return answer


# ============================================================
# OLLAMA
# ============================================================

def ask_ollama(
    user_message,
    language="en-ZA",
    memory_context="",
    conversation_context="",
    web_context="",
    document_context=""
):

    language_code = get_language_code(
        language
    )

    language_name = get_language_name(
        language
    )

    native_name = get_language_native_name(
        language
    )

    has_web = bool(
        web_context
    )

    has_document = bool(
        document_context
    )

    selected_model, route = choose_model(
        user_message,
        language_code,
        has_web,
        has_document
    )

    system_prompt = f"""
You are The Fix, a helpful general-purpose AI assistant.

{build_language_instructions(
    language_code,
    language_name,
    native_name
)}

RESPONSE STYLE:

- Answer the user's actual question.
- Be clear.
- Be direct.
- Be straightforward.
- Do not add unnecessary introductions.
- Do not repeat yourself.
- Never repeat the same sentence.
- Never repeat the same paragraph.
- Do not repeat the same idea unnecessarily.
- Do not repeat the user's question.
- Keep greetings very short.
- Keep simple questions short.
- Give more detail when the question requires it.
- For calculations, show the necessary working.
- For technical questions, explain clearly.
- Do not expose internal reasoning.
- Do not generate a long explanation for a simple greeting.
- Output only the final answer.
"""

    if language_code == "zu-ZA":

        system_prompt += """
FOR ISIZULU:

Respond naturally in isiZulu.

Use simple, understandable isiZulu.

Answer the user's actual request.

Do not refuse normal greetings.

Do not say that you cannot translate or answer unless
the request genuinely requires that.

Do not repeat sentences.

Do not repeat the question.

Do not switch to English unnecessarily.

English technical terms may be used when necessary.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    if memory_context:

        memory_text = str(
            memory_context
        )[:2000]

        messages.append({
            "role": "system",
            "content": (
                "Relevant memory:\n"
                + memory_text
            )
        })

    if conversation_context:

        messages.append({
            "role": "system",
            "content": (
                "Recent conversation:\n"
                + conversation_context
            )
        })

    if document_context:

        document_text = str(
            document_context
        )[:5000]

        messages.append({
            "role": "system",
            "content": (
                "Relevant document information:\n"
                + document_text
            )
        })

    if web_context:

        web_text = str(
            web_context
        )[:5000]

        messages.append({
            "role": "system",
            "content": (
                "Relevant current web information:\n"
                + web_text
            )
        })

    messages.append({
        "role": "user",
        "content": safe_string(
            user_message
        )
    })

    powerful = (
        selected_model == POWERFUL_MODEL
    )

    morena = (
        selected_model == MORENA_MODEL
    )

    if powerful:

        max_tokens = 280
        request_timeout = 120
        keep_alive = POWERFUL_KEEP_ALIVE

        temperature = 0.15
        repeat_penalty = 1.10
        repeat_last_n = 128

    elif morena:

        max_tokens = 150
        request_timeout = 60
        keep_alive = MORENA_KEEP_ALIVE

        temperature = 0.15
        repeat_penalty = 1.18
        repeat_last_n = 128

    else:

        max_tokens = 110
        request_timeout = 35
        keep_alive = FAST_KEEP_ALIVE

        temperature = 0.15
        repeat_penalty = 1.10
        repeat_last_n = 128

    payload = {
        "model": selected_model,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,

        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
        }
    }

    start_time = time.time()

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=request_timeout
        )

        elapsed = round(
            time.time() - start_time,
            2
        )

        if response.status_code != 200:

            return (
                (
                    "The Fix could not complete the request. "
                    f"Ollama returned HTTP "
                    f"{response.status_code}."
                ),
                selected_model,
                route,
                elapsed
            )

        data = response.json()

        message_data = data.get(
            "message",
            {}
        )

        answer = safe_string(
            message_data.get(
                "content",
                ""
            )
        )

        if not answer:

            answer = (
                "The Fix received an empty response."
            )

        answer = clean_repeated_response(
            answer
        )

        if not answer:

            answer = (
                "The Fix could not produce a final answer."
            )

        return (
            answer,
            selected_model,
            route,
            elapsed
        )

    except requests.exceptions.Timeout:

        return (
            (
                "The Fix is taking longer than expected. "
                "Please try the question again."
            ),
            selected_model,
            route,
            round(
                time.time() - start_time,
                2
            )
        )

    except requests.exceptions.ConnectionError:

        return (
            (
                "The Fix cannot connect to Ollama. "
                "Please make sure Ollama is running."
            ),
            selected_model,
            route,
            round(
                time.time() - start_time,
                2
            )
        )

    except Exception as e:

        return (
            f"The Fix encountered an error: {str(e)}",
            selected_model,
            route,
            round(
                time.time() - start_time,
                2
            )
        )


# ============================================================
# ROOT + FRONTEND
# ============================================================

@app.get("/")
def root():

    if FRONTEND_FILE.exists():

        return FileResponse(
            str(FRONTEND_FILE),
            media_type="text/html"
        )

    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "engine": "Ollama + SymPy",
        "model": OLLAMA_MODEL,
        "morena_model": MORENA_MODEL,
        "math_engine": "SymPy 1.14.0",
        "message": "The Fix API is running.",
        "frontend": "not found"
    }


@app.get("/api")
def api_root():

    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "engine": "Ollama + SymPy",
        "model": OLLAMA_MODEL,
        "morena_model": MORENA_MODEL,
        "math_engine": "SymPy 1.14.0",
        "message": "The Fix API is running."
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    models = get_available_models(
        force=True
    )

    return {
        "status": "online",
        "name": APP_NAME,
        "version": APP_VERSION,
        "ollama": True,
        "math_engine": True,
        "math_engine_name": "SymPy 1.14.0",
        "available_models": models,

        "fast_model": FAST_MODEL,
        "powerful_model": POWERFUL_MODEL,
        "vision_model": VISION_MODEL,
        "morena_model": MORENA_MODEL,

        "fast_available": FAST_MODEL in models,
        "powerful_available": POWERFUL_MODEL in models,
        "vision_available": VISION_MODEL in models,
        "morena_available": MORENA_MODEL in models
    }


# ============================================================
# MODELS
# ============================================================

@app.get("/models")
def models():

    available = get_available_models(
        force=True
    )

    return {
        "success": True,

        "fast_model": FAST_MODEL,
        "powerful_model": POWERFUL_MODEL,
        "vision_model": VISION_MODEL,
        "morena_model": MORENA_MODEL,

        "available_models": available,

        "fast_available": FAST_MODEL in available,
        "powerful_available": POWERFUL_MODEL in available,
        "vision_available": VISION_MODEL in available,
        "morena_available": MORENA_MODEL in available
    }


# ============================================================
# LANGUAGES
# ============================================================

@app.get("/languages")
def languages():

    return {
        "success": True,
        "languages": LANGUAGES,
        "native_names": LANGUAGE_NATIVE_NAMES,
        "multilingual_languages": list(
            MULTILINGUAL_LANGUAGES
        ),
        "african_languages": list(
            AFRICAN_LANGUAGES
        )
    }


# ============================================================
# REGISTER
# ============================================================

@app.post("/register")
def register(
    request: RegisterRequest
):

    username = safe_string(
        request.username
    )

    password = safe_string(
        request.password
    )

    if not username:

        raise HTTPException(
            status_code=400,
            detail="Username is required."
        )

    if not password:

        raise HTTPException(
            status_code=400,
            detail="Password is required."
        )

    try:

        return create_user(
            username,
            password
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ============================================================
# LOGIN
# ============================================================

@app.post("/login")
def login(
    request: LoginRequest
):

    try:

        result = authenticate_user(
            request.username,
            request.password
        )

        if not result:

            raise HTTPException(
                status_code=401,
                detail="Invalid username or password."
            )

        return result

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# USER
# ============================================================

@app.get("/user/{user_id}")
def user(
    user_id: str
):

    try:

        result = get_user(
            user_id
        )

        if not result:

            raise HTTPException(
                status_code=404,
                detail="User not found."
            )

        return result

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# MEMORY
# ============================================================

@app.post("/memory")
def save_memory(
    request: MemoryRequest
):

    try:

        result = remember(
            request.user_id,
            request.key,
            request.value
        )

        return {
            "success": True,
            "result": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/memory/{user_id}")
def memory(
    user_id: str
):

    try:

        return {
            "success": True,
            "user_id": user_id,
            "memories": get_memories(
                user_id
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/memory-summary/{user_id}")
def memory_summary(
    user_id: str
):

    try:

        return {
            "success": True,
            "user_id": user_id,
            "summary": get_memory_summary(
                user_id
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/forget")
def forget_memory(
    request: ForgetRequest
):

    try:

        result = forget(
            request.user_id,
            request.key
        )

        return {
            "success": True,
            "result": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# CONVERSATION
# ============================================================

@app.get("/conversation/{user_id}")
def conversation(
    user_id: str
):

    try:

        return {
            "success": True,
            "user_id": user_id,
            "conversation": get_conversation_history(
                user_id
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/conversation/clear")
def clear_conversation(
    request: ConversationClearRequest
):

    try:

        result = clear_conversation_history(
            request.user_id
        )

        return {
            "success": True,
            "result": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
def chat(
    request: ChatRequest
):

    start_time = time.time()

    user_id = safe_string(
        request.user_id
    )

    message = safe_string(
        request.message
    )

    requested_language = get_language_code(
        request.language
    )

    if not user_id:

        raise HTTPException(
            status_code=400,
            detail="user_id is required."
        )

    if not message:

        raise HTTPException(
            status_code=400,
            detail="message is required."
        )

    # ========================================================
    # AUTOMATIC INPUT LANGUAGE DETECTION
    # ========================================================

    language = detect_input_language(
        message,
        requested_language
    )

    # ========================================================
    # BASIC CALCULATOR
    # ========================================================

    try:

        calculator_result = use_calculator(
            message
        )

        if calculator_result is not None:

            answer = str(
                calculator_result
            )

            try:

                add_conversation(
                    user_id,
                    message,
                    answer
                )

            except Exception:
                pass

            return {
                "name": APP_NAME,
                "version": APP_VERSION,
                "user_id": user_id,
                "message": message,
                "language": language,
                "language_name": get_language_name(
                    language
                ),
                "language_native_name": get_language_native_name(
                    language
                ),
                "engine": "Calculator",
                "model": "calculator",
                "route": "calculator",
                "web_search": False,
                "memory_saved": False,
                "response_time": round(
                    time.time() - start_time,
                    3
                ),
                "answer": answer
            }

    except Exception:
        pass

    # ========================================================
    # ADVANCED MATHEMATICS ENGINE
    # ========================================================

    try:

        math_result = solve_math(
            message
        )

        if math_result is not None:

            math_answer = math_result.get(
                "answer",
                ""
            )

            math_steps = math_result.get(
                "steps",
                []
            )

            if math_steps:

                answer_parts = []

                for step in math_steps:

                    answer_parts.append(
                        str(step)
                    )

                answer = "\n".join(
                    answer_parts
                )

            else:

                answer = str(
                    math_answer
                )

            try:

                add_conversation(
                    user_id,
                    message,
                    answer
                )

            except Exception:
                pass

            return {
                "name": APP_NAME,
                "version": APP_VERSION,
                "user_id": user_id,
                "message": message,
                "language": language,
                "language_name": get_language_name(
                    language
                ),
                "language_native_name": get_language_native_name(
                    language
                ),
                "engine": "SymPy",
                "model": "sympy-1.14.0",
                "route": "advanced-mathematics",
                "web_search": False,
                "memory_saved": False,
                "model_response_time": 0.0,
                "total_response_time": round(
                    time.time() - start_time,
                    3
                ),
                "answer": answer
            }

    except Exception:
        pass

    # ========================================================
    # GREETINGS
    # ========================================================

    greeting = detect_greeting(
        message
    )

    if greeting:

        answer = greeting_response(
            language
        )

        try:

            add_conversation(
                user_id,
                message,
                answer
            )

        except Exception:
            pass

        return {
            "name": APP_NAME,
            "version": APP_VERSION,
            "user_id": user_id,
            "message": message,
            "language": language,
            "language_name": get_language_name(
                language
            ),
            "language_native_name": get_language_native_name(
                language
            ),
            "engine": "The Fix",
            "model": "local-response",
            "route": "greeting",
            "web_search": False,
            "memory_saved": False,
            "model_response_time": 0.0,
            "total_response_time": round(
                time.time() - start_time,
                3
            ),
            "answer": answer
        }

    # ========================================================
    # MEMORY
    # ========================================================

    memory_saved = detect_and_save_memory(
        user_id,
        message
    )

    memory_context = get_memory_context(
        user_id
    )

    conversation_context = get_conversation_context(
        user_id
    )

    # ========================================================
    # WEB
    # ========================================================

    web_context = ""
    web_used = False
    web_sources = []

    if needs_web_search(
        message
    ):

        try:

            web_result = web_search(
                message
            )

            if web_result:

                web_used = True

                if isinstance(
                    web_result,
                    dict
                ):

                    web_context = str(
                        web_result.get(
                            "answer",
                            web_result
                        )
                    )

                    web_sources = web_result.get(
                        "sources",
                        []
                    )

                else:

                    web_context = str(
                        web_result
                    )

        except Exception:

            web_context = ""

    # ========================================================
    # ASK MODEL
    # ========================================================

    answer, selected_model, route, model_time = ask_ollama(
        user_message=message,
        language=language,
        memory_context=memory_context,
        conversation_context=conversation_context,
        web_context=web_context,
        document_context=""
    )

    # ========================================================
    # MEMORY CONFIRMATION
    # ========================================================

    if memory_saved:

        confirmations = {

            "en-ZA":
                " I saved that to your memory.",

            "en-US":
                " I saved that to your memory.",

            "zu-ZA":
                " Ngikugcine lokho enkumbulweni yakho.",

            "xh-ZA":
                " Ndikugcine oko kwinkumbulo yakho.",

            "st-ZA":
                " Ke e bolokile seo mohopolong wa hao.",

            "af-ZA":
                " Ek het dit in jou geheue gestoor.",

            "tn-ZA":
                " Ke go bolokile seo mo kgopolong ya gago.",

            "nso-ZA":
                " Ke e bolokile mo kgopolong ya gago.",

            "ss-ZA":
                " Ngikugcine loko enkumbulweni yakho.",

            "ts-ZA":
                " Ndza swi hlayisa eka miehleketo ya wena.",

            "ve-ZA":
                " Ndo zwi vhulunga kha muhumbulo waṋu.",

            "nr-ZA":
                " Ngikugcine lokho enkumbulweni yakho."
        }

        answer += confirmations.get(
            language,
            " I saved that to your memory."
        )

    # ========================================================
    # WEB SOURCES
    # ========================================================

    if web_used and web_sources:

        if isinstance(
            web_sources,
            list
        ):

            source_text = "\n\nSources:\n"

            for source in web_sources[:5]:

                source_text += (
                    f"- {source}\n"
                )

            answer += source_text

    # ========================================================
    # FINAL CLEANUP
    # ========================================================

    answer = clean_repeated_response(
        answer
    )

    # ========================================================
    # SAVE CONVERSATION
    # ========================================================

    try:

        add_conversation(
            user_id,
            message,
            answer
        )

    except Exception:
        pass

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "user_id": user_id,
        "message": message,
        "language": language,
        "language_name": get_language_name(
            language
        ),
        "language_native_name": get_language_native_name(
            language
        ),
        "engine": "Ollama",
        "model": selected_model,
        "route": route,
        "web_search": web_used,
        "memory_saved": memory_saved,
        "model_response_time": model_time,
        "total_response_time": round(
            time.time() - start_time,
            2
        ),
        "answer": answer
    }


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...)
):

    if read_document is None:

        raise HTTPException(
            status_code=500,
            detail="Document reader is not available."
        )

    try:

        content = await file.read()

        if not content:

            raise HTTPException(
                status_code=400,
                detail="The uploaded document is empty."
            )

        documents_folder = os.path.join(
            os.path.dirname(__file__),
            "documents"
        )

        os.makedirs(
            documents_folder,
            exist_ok=True
        )

        filename = safe_string(
            file.filename
        )

        if not filename:

            raise HTTPException(
                status_code=400,
                detail="Invalid filename."
            )

        safe_filename = os.path.basename(
            filename
        )

        file_path = os.path.join(
            documents_folder,
            f"{int(time.time() * 1000)}_{safe_filename}"
        )

        with open(
            file_path,
            "wb"
        ) as f:

            f.write(content)

        try:

            result = read_document(
                file_path
            )

        except TypeError:

            result = read_document(
                file_path,
                safe_filename
            )

        return {
            "success": True,
            "filename": os.path.basename(
                file_path
            ),
            "original_filename": safe_filename,
            "path": file_path,
            "result": result
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {str(e)}"
        )


# ============================================================
# DOCUMENT LIST
# ============================================================

@app.get("/documents")
def documents():

    documents_folder = os.path.join(
        os.path.dirname(__file__),
        "documents"
    )

    os.makedirs(
        documents_folder,
        exist_ok=True
    )

    files = []

    for filename in os.listdir(
        documents_folder
    ):

        path = os.path.join(
            documents_folder,
            filename
        )

        if os.path.isfile(
            path
        ):

            files.append({
                "filename": filename,
                "size": os.path.getsize(
                    path
                )
            })

    return {
        "success": True,
        "documents": files
    }


# ============================================================
# DOCUMENT SEARCH
# ============================================================

@app.post("/document-search")
def document_search(
    request: DocumentSearchRequest
):

    if search_document is None:

        raise HTTPException(
            status_code=500,
            detail="Document search is not available."
        )

    try:

        documents_folder = os.path.join(
            os.path.dirname(__file__),
            "documents"
        )

        file_path = os.path.join(
            documents_folder,
            os.path.basename(
                request.filename
            )
        )

        if not os.path.exists(
            file_path
        ):

            raise HTTPException(
                status_code=404,
                detail="Document not found."
            )

        try:

            result = search_document(
                file_path,
                request.query,
                request.top_k
            )

        except TypeError:

            result = search_document(
                file_path,
                request.query
            )

        return {
            "success": True,
            "filename": request.filename,
            "query": request.query,
            "results": result
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Document search failed: {str(e)}"
        )


# ============================================================
# DOCUMENT QUESTION
# ============================================================

@app.post("/document-question")
def document_question(
    request: DocumentQuestionRequest
):

    if search_document is None:

        raise HTTPException(
            status_code=500,
            detail="Document search is not available."
        )

    documents_folder = os.path.join(
        os.path.dirname(__file__),
        "documents"
    )

    file_path = os.path.join(
        documents_folder,
        os.path.basename(
            request.filename
        )
    )

    if not os.path.exists(
        file_path
    ):

        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    try:

        try:

            results = search_document(
                file_path,
                request.question,
                4
            )

        except TypeError:

            results = search_document(
                file_path,
                request.question
            )

        if isinstance(
            results,
            list
        ):

            evidence_parts = []

            for item in results[:4]:

                if isinstance(
                    item,
                    dict
                ):

                    text = (
                        item.get("text")
                        or item.get("content")
                        or str(item)
                    )

                else:

                    text = str(item)

                evidence_parts.append(
                    str(text)[:1400]
                )

            document_context = (
                "\n\n---\n\n".join(
                    evidence_parts
                )
            )

        else:

            document_context = str(
                results
            )[:5000]

        detected_language = detect_input_language(
            request.question,
            request.language
        )

        answer, selected_model, route, model_time = ask_ollama(
            user_message=request.question,
            language=detected_language,
            memory_context=get_memory_context(
                request.user_id
            ),
            conversation_context=get_conversation_context(
                request.user_id
            ),
            web_context="",
            document_context=document_context
        )

        answer = clean_repeated_response(
            answer
        )

        try:

            add_conversation(
                request.user_id,
                request.question,
                answer
            )

        except Exception:
            pass

        return {
            "success": True,
            "name": APP_NAME,
            "version": APP_VERSION,
            "filename": request.filename,
            "language": detected_language,
            "language_name": get_language_name(
                detected_language
            ),
            "language_native_name": get_language_native_name(
                detected_language
            ),
            "model": selected_model,
            "route": route,
            "model_response_time": model_time,
            "answer": answer
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Document question failed: {str(e)}"
        )


# ============================================================
# LEGACY PDF ROUTES
# ============================================================

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...)
):

    return await upload_document(
        file
    )


@app.post("/pdf-search")
def pdf_search(
    request: DocumentSearchRequest
):

    return document_search(
        request
    )


@app.post("/pdf-question")
def pdf_question(
    request: DocumentQuestionRequest
):

    return document_question(
        request
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    print("=" * 60)
    print("THE FIX")
    print(f"Version: {APP_VERSION}")
    print("=" * 60)
    print(f"Fast model:     {FAST_MODEL}")
    print(f"Powerful model: {POWERFUL_MODEL}")
    print(f"Vision model:   {VISION_MODEL}")
    print(f"isiZulu model:  {MORENA_MODEL}")
    print("Math engine:    SymPy 1.14.0")
    print("=" * 60)
    print("Speed optimization: ENABLED")
    print("Response optimization: ENABLED")
    print("Anti-repetition: ENABLED")
    print("Automatic language detection: ENABLED")
    print("Greeting optimization: ENABLED")
    print("Advanced mathematics: ENABLED")
    print("Fast model keep-alive: 2h")
    print("MORENA keep-alive: 2h")
    print("=" * 60)
    print("Starting The Fix API...")
    print("=" * 60)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )