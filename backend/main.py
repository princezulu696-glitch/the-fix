import os
import time
import re
import requests

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
# OPTIONAL DOCUMENT MODULES
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
# CONFIGURATION
# ============================================================

load_dotenv()

APP_NAME = "The Fix"

APP_VERSION = "8.0.0"

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

OLLAMA_MODEL = "qwen3:1.7b"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="The Fix API",
    version=APP_VERSION
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):
    user_id: str
    message: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class MemoryRequest(BaseModel):
    user_id: str
    key: str
    value: str
    category: str = "general"


class ForgetRequest(BaseModel):
    user_id: str
    key: str


class ClearConversationRequest(BaseModel):
    user_id: str


class PDFQuestionRequest(BaseModel):
    user_id: str
    question: str
    filename: str


class PDFSearchRequest(BaseModel):
    user_id: str
    query: str
    filename: str


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_string(value):

    if value is None:
        return ""

    return str(value).strip()


def clean_answer(text):

    if not text:
        return ""

    text = str(text).strip()

    # Remove hidden thinking blocks.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    text = text.replace(
        "<|im_end|>",
        ""
    )

    return text.strip()


# ============================================================
# OLLAMA STATUS
# ============================================================

def ollama_available():

    try:

        response = requests.get(
            "http://127.0.0.1:11434/api/tags",
            timeout=5
        )

        return response.status_code == 200

    except Exception:

        return False


# ============================================================
# MEMORY FORMATTER
# ============================================================

def build_memory_text(user_id):

    try:

        memories = get_memories(
            user_id
        )

    except Exception:

        return ""


    if not memories:

        return ""


    lines = []


    if isinstance(
        memories,
        dict
    ):

        for key, value in memories.items():

            if value is None:
                continue

            key = safe_string(key)
            value = safe_string(value)

            if key and value:

                lines.append(
                    f"- {key}: {value}"
                )


    elif isinstance(
        memories,
        list
    ):

        for item in memories:

            if isinstance(
                item,
                dict
            ):

                key = safe_string(
                    item.get(
                        "key",
                        ""
                    )
                )

                value = safe_string(
                    item.get(
                        "value",
                        ""
                    )
                )

                category = safe_string(
                    item.get(
                        "category",
                        "general"
                    )
                )


                if key and value:

                    lines.append(
                        f"- [{category}] {key}: {value}"
                    )

            else:

                lines.append(
                    f"- {safe_string(item)}"
                )


    return "\n".join(lines)


# ============================================================
# CONVERSATION FORMATTER
# ============================================================

def build_conversation_text(user_id):

    try:

        history = get_conversation_history(
            user_id
        )

    except Exception:

        return ""


    if not history:

        return ""


    lines = []


    for item in history[-10:]:

        if not isinstance(
            item,
            dict
        ):

            continue


        role = safe_string(
            item.get(
                "role",
                ""
            )
        )


        content = safe_string(
            item.get(
                "content",
                item.get(
                    "message",
                    ""
                )
            )
        )


        if role and content:

            lines.append(
                f"{role}: {content}"
            )


    return "\n".join(lines)


# ============================================================
# AUTOMATIC MEMORY DETECTION
# ============================================================

def detect_automatic_memory(message):

    text = safe_string(
        message
    )

    lower = text.lower().strip()


    if not lower:

        return None


    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    patterns = [

        r"^remember that my name is\s+(.+)$",

        r"^my name is\s+(.+)$",

        r"^remember my name is\s+(.+)$",

        r"^call me\s+(.+)$",

        r"^from now on, call me\s+(.+)$",

        r"^remember to call me\s+(.+)$",

    ]


    for pattern in patterns:

        match = re.match(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).strip()

            value = re.sub(
                r"[.!?]+$",
                "",
                value
            ).strip()

            if value:

                return {
                    "key": "name",
                    "value": value,
                    "category": "profile"
                }


    # --------------------------------------------------------
    # FAVORITE COLOR
    # --------------------------------------------------------

    patterns = [

        r"^remember that my favorite color is\s+(.+)$",

        r"^my favorite color is\s+(.+)$",

        r"^my favourite color is\s+(.+)$",

        r"^remember my favorite color is\s+(.+)$",

        r"^remember my favourite colour is\s+(.+)$",

    ]


    for pattern in patterns:

        match = re.match(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).strip()

            value = re.sub(
                r"[.!?]+$",
                "",
                value
            ).strip()

            if value:

                return {
                    "key": "favorite color",
                    "value": value,
                    "category": "preference"
                }


    # --------------------------------------------------------
    # FAVORITE SUBJECT
    # --------------------------------------------------------

    patterns = [

        r"^remember that my favorite subject is\s+(.+)$",

        r"^my favorite subject is\s+(.+)$",

        r"^my favourite subject is\s+(.+)$",

        r"^remember my favorite subject is\s+(.+)$",

        r"^remember my favourite subject is\s+(.+)$",

    ]


    for pattern in patterns:

        match = re.match(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).strip()

            value = re.sub(
                r"[.!?]+$",
                "",
                value
            ).strip()

            if value:

                return {
                    "key": "favorite subject",
                    "value": value,
                    "category": "preference"
                }


    # --------------------------------------------------------
    # PREFERENCE
    # --------------------------------------------------------

    patterns = [

        r"^remember that i prefer\s+(.+)$",

        r"^i prefer\s+(.+)$",

        r"^remember that i like\s+(.+)$",

        r"^i like\s+(.+)$",

    ]


    for pattern in patterns:

        match = re.match(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).strip()

            value = re.sub(
                r"[.!?]+$",
                "",
                value
            ).strip()

            if value:

                return {
                    "key": "preference",
                    "value": value,
                    "category": "preference"
                }


    # --------------------------------------------------------
    # CURRENT PROJECT
    # --------------------------------------------------------

    patterns = [

        r"^remember that i am working on\s+(.+)$",

        r"^i am working on\s+(.+)$",

        r"^i'm working on\s+(.+)$",

        r"^remember that i'm working on\s+(.+)$",

        r"^my current project is\s+(.+)$",

        r"^remember that my current project is\s+(.+)$",

        r"^i am building\s+(.+)$",

        r"^i'm building\s+(.+)$",

        r"^remember that i am building\s+(.+)$",

    ]


    for pattern in patterns:

        match = re.match(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).strip()

            value = re.sub(
                r"[.!?]+$",
                "",
                value
            ).strip()

            # Remove temporary time words.
            value = re.sub(
                r"\s+(today|tonight|this week|this month)$",
                "",
                value,
                flags=re.IGNORECASE
            ).strip()

            if value:

                return {
                    "key": "current project",
                    "value": value,
                    "category": "project"
                }


    # --------------------------------------------------------
    # FAVORITE PROGRAMMING LANGUAGE
    # --------------------------------------------------------

    patterns = [

        r"^my favorite programming language is\s+(.+)$",

        r"^my favourite programming language is\s+(.+)$",

        r"^remember that my favorite programming language is\s+(.+)$",

        r"^remember that my favourite programming language is\s+(.+)$",

    ]


    for pattern in patterns:

        match = re.match(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).strip()

            value = re.sub(
                r"[.!?]+$",
                "",
                value
            ).strip()

            if value:

                return {
                    "key": "favorite programming language",
                    "value": value,
                    "category": "preference"
                }


    return None


# ============================================================
# AUTOMATIC MEMORY SAVER
# ============================================================

def save_automatic_memory(
    user_id,
    message
):

    memory = detect_automatic_memory(
        message
    )


    if not memory:

        return None


    try:

        remember(

            user_id,

            memory["key"],

            memory["value"]

        )

        return memory


    except Exception:

        return None


# ============================================================
# SMART WEB SEARCH DETECTION
# ============================================================

CURRENT_QUESTION_PATTERNS = [

    r"\bwhat is the latest\b",

    r"\bwhat's the latest\b",

    r"\blatest news\b",

    r"\brecent news\b",

    r"\bwhat happened today\b",

    r"\bwhat happened yesterday\b",

    r"\bwhat is happening now\b",

    r"\bwhat's happening now\b",

    r"\bwhat is happening currently\b",

    r"\bwhat's happening currently\b",

    r"\bcurrent price\b",

    r"\bcurrent weather\b",

    r"\bcurrent president\b",

    r"\bcurrent news\b",

    r"\bcurrent events\b",

    r"\bwho is currently\b",

    r"\bwhat is currently\b",

    r"\bwhat's currently\b",

    r"\bsearch the web\b",

    r"\bsearch online\b",

    r"\blook it up\b",

    r"\bfind online\b",

    r"\baccording to recent\b",

    r"\bthis week's news\b",

    r"\bthis month's news\b",

]


def needs_web_search(message):

    text = safe_string(
        message
    ).lower()


    if not text:

        return False


    for pattern in CURRENT_QUESTION_PATTERNS:

        if re.search(
            pattern,
            text
        ):

            return True


    # Direct current-information questions.
    current_question_patterns = [

        r"^what is .* now\??$",

        r"^what's .* now\??$",

        r"^who is .* now\??$",

        r"^where is .* now\??$",

        r"^how much is .* now\??$",

        r"^how much does .* cost now\??$",

        r"^is .* available now\??$",

    ]


    for pattern in current_question_patterns:

        if re.search(
            pattern,
            text
        ):

            return True


    return False


# ============================================================
# WEB EVIDENCE FORMATTER
# ============================================================

def format_web_sources(results):

    if not results:

        return []


    formatted = []


    for result in results:

        formatted.append({

            "source":
                result.get(
                    "source",
                    ""
                ),

            "title":
                result.get(
                    "title",
                    ""
                ),

            "url":
                result.get(
                    "url",
                    ""
                ),

            "text":
                result.get(
                    "text",
                    ""
                )
        })


    return formatted


# ============================================================
# SOURCE SECTION
# ============================================================

def build_source_section(results):

    if not results:

        return ""


    lines = [
        "",
        "Sources:"
    ]


    number = 1


    for result in results:

        source = safe_string(
            result.get(
                "source",
                ""
            )
        )

        title = safe_string(
            result.get(
                "title",
                ""
            )
        )

        url = safe_string(
            result.get(
                "url",
                ""
            )
        )


        if not url:

            continue


        lines.append(
            f"{number}. {source} — {title}"
        )

        lines.append(
            f"   {url}"
        )


        number += 1


    return "\n".join(lines)


# ============================================================
# MAIN AI FUNCTION
# ============================================================

def ask_ollama(
    user_message,
    memory_text="",
    conversation_text="",
    web_evidence=None,
    document_evidence=""
):

    web_evidence = web_evidence or []


    system_prompt = """
You are The Fix.

Your name is The Fix.

You are an independent AI assistant.

You must be helpful, accurate, clear and honest.

============================================================
MEMORY RULES
============================================================

USER MEMORY contains information that The Fix has saved about
the user.

You MUST use USER MEMORY when answering questions about the
user.

If the user asks:

"What is my favorite color?"

and USER MEMORY says:

favorite color: blue

then answer:

"Your favorite color is blue."

Do NOT say:

"I don't have access to personal information about you."

Do NOT ignore USER MEMORY.

If USER MEMORY contains the answer, use it directly.

If USER MEMORY does not contain the answer, honestly say that
you do not have that information saved.

============================================================
CONVERSATION RULES
============================================================

Use RECENT CONVERSATION to understand what the user is talking
about.

Give priority to the most recent user statement when the user
asks what they just said.

Do not invent previous conversations.

============================================================
GENERAL AI RULES
============================================================

1. Your name is The Fix.

2. Never claim to be human.

3. Never falsely claim to have human consciousness.

4. Never falsely claim to have actual human feelings.

5. You can understand emotional context and respond with
   empathy.

6. Do not invent facts.

7. Do not invent sources.

8. Do not invent URLs.

9. Answer normal questions using your model knowledge.

10. For mathematical questions, provide the correct result.

11. Keep answers reasonably concise unless the user asks for
    more detail.

============================================================
CURRENT INFORMATION RULE
============================================================

When CURRENT WEB EVIDENCE is provided, use that evidence for
current-information questions.

Do not invent current events that are not supported by the
provided evidence.

If the evidence is insufficient, say that the available web
evidence is insufficient.

============================================================
DOCUMENT RULE
============================================================

When DOCUMENT EVIDENCE is provided, use it to answer questions
about the document.

Do not invent information that is not supported by the document.

============================================================
IMPORTANT
============================================================

USER MEMORY is information about the user.

Use it.

Do not ignore it.

Do not claim that you cannot access it when it is explicitly
provided below.
"""


    messages = [

        {
            "role": "system",
            "content": system_prompt
        }

    ]


    # ========================================================
    # USER MEMORY
    # ========================================================

    if memory_text:

        messages.append({

            "role": "system",

            "content":
                """
USER MEMORY:

The following information is saved about this user.

Use this information when it is relevant.

Do not deny access to this information.

"""
                + memory_text
        })


    # ========================================================
    # RECENT CONVERSATION
    # ========================================================

    if conversation_text:

        messages.append({

            "role": "system",

            "content":
                "RECENT CONVERSATION:\n"
                + conversation_text
        })


    # ========================================================
    # DOCUMENT EVIDENCE
    # ========================================================

    if document_evidence:

        messages.append({

            "role": "system",

            "content":
                """
DOCUMENT EVIDENCE:

Use the following document information to answer the user's
document question.

"""
                + document_evidence
        })


    # ========================================================
    # WEB EVIDENCE
    # ========================================================

    if web_evidence:

        evidence_lines = []


        for index, result in enumerate(
            web_evidence,
            start=1
        ):

            source = safe_string(
                result.get(
                    "source",
                    "Unknown source"
                )
            )

            title = safe_string(
                result.get(
                    "title",
                    ""
                )
            )

            url = safe_string(
                result.get(
                    "url",
                    ""
                )
            )

            text = safe_string(
                result.get(
                    "text",
                    ""
                )
            )


            evidence_lines.append(
                f"""
SOURCE {index}
Source: {source}
Title: {title}
URL: {url}
Evidence: {text}
"""
            )


        web_block = "\n".join(
            evidence_lines
        )


        messages.append({

            "role": "system",

            "content":
                """
CURRENT WEB EVIDENCE:

Use the supplied evidence when answering current-information
questions.

Do not invent additional current facts.

"""
                + web_block
        })


    # ========================================================
    # USER MESSAGE
    # ========================================================

    messages.append({

        "role": "user",

        "content":
            safe_string(user_message)
    })


    # ========================================================
    # OLLAMA REQUEST
    # ========================================================

    payload = {

        "model":
            OLLAMA_MODEL,

        "messages":
            messages,

        "stream":
            False,

        "think":
            False,

        "options": {

            "temperature":
                0.1
        }
    }


    try:

        response = requests.post(

            OLLAMA_URL,

            json=payload,

            timeout=180
        )


        response.raise_for_status()


        data = response.json()


        answer = (

            data
            .get(
                "message",
                {}
            )
            .get(
                "content",
                ""
            )
        )


        answer = clean_answer(
            answer
        )


        if not answer:

            return (
                "The Fix could not generate an answer."
            )


        return answer


    except requests.exceptions.Timeout:

        return (
            "The local AI model took too long to respond. "
            "Please try again."
        )


    except requests.exceptions.ConnectionError:

        return (
            "The local Ollama service is not available. "
            "Please make sure Ollama is running."
        )


    except Exception as error:

        return (
            "The Fix could not generate an answer right now. "
            f"Error: {error}"
        )


# ============================================================
# WEB ANSWER CLEANUP
# ============================================================

def validate_web_answer(
    answer,
    web_results
):

    if not answer:

        return (
            "I could not generate a reliable answer from "
            "the available web evidence."
        )


    if not web_results:

        return answer


    suspicious_phrases = [

        "i browsed the internet",

        "i personally checked",

        "i searched the internet",

        "according to my knowledge today"
    ]


    for phrase in suspicious_phrases:

        answer = re.sub(

            re.escape(phrase),

            "",

            answer,

            flags=re.IGNORECASE
        )


    return answer.strip()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "name":
            APP_NAME,

        "version":
            APP_VERSION,

        "status":
            "online",

        "engine":
            "Ollama",

        "model":
            OLLAMA_MODEL,

        "message":
            "The Fix API is running."
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "name":
            APP_NAME,

        "version":
            APP_VERSION,

        "api":
            "online",

        "ollama":
            ollama_available(),

        "model":
            OLLAMA_MODEL
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

            "success":
                True,

            "user":
                user
        }


    except ValueError as error:

        raise HTTPException(

            status_code=400,

            detail=str(error)
        )


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)
        )


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


    if not user:

        raise HTTPException(

            status_code=401,

            detail="Invalid username or password."
        )


    return {

        "success":
            True,

        "user":
            user
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


    if not user:

        raise HTTPException(

            status_code=404,

            detail="User not found."
        )


    return user


# ============================================================
# SAVE MEMORY
# ============================================================

@app.post("/memory")
def save_user_memory(
    request: MemoryRequest
):

    try:

        result = remember(

            request.user_id,

            request.key,

            request.value
        )


        return {

            "success":
                True,

            "memory":
                result
        }


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)
        )


# ============================================================
# READ MEMORY
# ============================================================

@app.get("/memory/{user_id}")
def read_user_memory(
    user_id: str
):

    try:

        return {

            "user_id":
                user_id,

            "memories":
                get_memories(
                    user_id
                )
        }


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)
        )


# ============================================================
# MEMORY SUMMARY
# ============================================================

@app.get("/memory-summary/{user_id}")
def memory_summary(
    user_id: str
):

    try:

        return get_memory_summary(
            user_id
        )


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)
        )


# ============================================================
# FORGET MEMORY
# ============================================================

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

            "success":
                True,

            "result":
                result
        }


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)
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

            "user_id":
                user_id,

            "conversation":
                get_conversation_history(
                    user_id
                )
        }


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)
        )


# ============================================================
# CLEAR CONVERSATION
# ============================================================

@app.post("/conversation/clear")
def clear_conversation(
    request: ClearConversationRequest
):

    try:

        result = clear_conversation_history(

            request.user_id
        )


        return {

            "success":
                True,

            "result":
                result
        }


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)
        )


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
def chat(
    request: ChatRequest
):

    user_id = safe_string(
        request.user_id
    )

    message = safe_string(
        request.message
    )


    if not message:

        raise HTTPException(

            status_code=400,

            detail="Message cannot be empty."
        )


    # ========================================================
    # CALCULATOR
    # ========================================================

    calculator_result = use_calculator(
        message
    )


    if calculator_result:

        answer = (

            f"The answer is "
            f"{calculator_result['result']}."
        )


        try:

            add_conversation(

                user_id,

                "user",

                message
            )


            add_conversation(

                user_id,

                "assistant",

                answer
            )

        except Exception:

            pass


        return {

            "name":
                APP_NAME,

            "user_id":
                user_id,

            "message":
                message,

            "tool":
                "calculator",

            "expression":
                calculator_result[
                    "expression"
                ],

            "answer":
                answer
        }


    # ========================================================
    # AUTOMATIC MEMORY
    # ========================================================

    automatic_memory = save_automatic_memory(

        user_id,

        message
    )


    # ========================================================
    # LOAD MEMORY
    # ========================================================

    memory_text = build_memory_text(
        user_id
    )


    # ========================================================
    # LOAD CONVERSATION
    # ========================================================

    conversation_text = (
        build_conversation_text(
            user_id
        )
    )


    # ========================================================
    # WEB SEARCH
    # ========================================================

    web_results = []


    if needs_web_search(
        message
    ):

        search_result = web_search(
            message
        )


        if search_result.get(
            "success",
            False
        ):

            web_results = (

                search_result.get(

                    "results",

                    []
                )
            )


    # ========================================================
    # ASK THE FIX
    # ========================================================

    answer = ask_ollama(

        user_message=
            message,

        memory_text=
            memory_text,

        conversation_text=
            conversation_text,

        web_evidence=
            format_web_sources(
                web_results
            )
    )


    # ========================================================
    # MEMORY CONFIRMATION
    # ========================================================

    if automatic_memory:

        key = automatic_memory.get(
            "key",
            ""
        )

        value = automatic_memory.get(
            "value",
            ""
        )


        if key and value:

            answer = (

                f"{answer}\n\n"
                f"I've saved that: {key} = {value}."
            )


    # ========================================================
    # WEB CLEANUP
    # ========================================================

    if web_results:

        answer = validate_web_answer(

            answer,

            web_results
        )


        answer += build_source_section(
            web_results
        )


    # ========================================================
    # SAVE CONVERSATION
    # ========================================================

    try:

        add_conversation(

            user_id,

            "user",

            message
        )


        add_conversation(

            user_id,

            "assistant",

            answer
        )

    except Exception:

        pass


    # ========================================================
    # RESPONSE
    # ========================================================

    response = {

        "name":
            APP_NAME,

        "version":
            APP_VERSION,

        "user_id":
            user_id,

        "message":
            message,

        "engine":
            "Ollama",

        "model":
            OLLAMA_MODEL,

        "web_search":
            bool(web_results),

        "memory_saved":
            bool(automatic_memory),

        "answer":
            answer
    }


    if automatic_memory:

        response["saved_memory"] = {

            "key":
                automatic_memory.get(
                    "key",
                    ""
                ),

            "value":
                automatic_memory.get(
                    "value",
                    ""
                ),

            "category":
                automatic_memory.get(
                    "category",
                    "general"
                )
        }


    if web_results:

        response["sources"] = [

            {

                "source":
                    item.get(
                        "source",
                        ""
                    ),

                "title":
                    item.get(
                        "title",
                        ""
                    ),

                "url":
                    item.get(
                        "url",
                        ""
                    )
            }

            for item in web_results
        ]


    return response


# ============================================================
# PDF UPLOAD
# ============================================================

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File name is missing."
        )

    allowed_extensions = {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".txt"
    }

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported document type. "
                "Supported files: PDF, DOCX, PPTX, XLSX and TXT."
            )
        )

    if read_document is None:
        raise HTTPException(
            status_code=500,
            detail="Document reader is not available."
        )

    try:
        content = await file.read()

        safe_filename = os.path.basename(
            file.filename
        )

        stored_filename = (
            str(int(time.time() * 1000))
            + "_"
            + safe_filename
        )

        temp_filename = os.path.join(
            "documents",
            stored_filename
        )

        with open(
            temp_filename,
            "wb"
        ) as output:
            output.write(content)

        text = read_document(
            temp_filename
        )

        return {
    "success": True,
    "filename": file.filename,
    "stored_filename": stored_filename,
    "type": extension,
    "characters": len(text),
    "stored_filename": stored_filename,
    "text": text
    
}

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
# ============================================================
# PDF SEARCH
# ============================================================

@app.get("/documents")
def list_documents():

    documents_folder = "documents"

    if not os.path.exists(documents_folder):
        return {
            "success": True,
            "documents": []
        }

    document_files = os.listdir(
        documents_folder
    )

    documents = []

    for filename in document_files:

        file_path = os.path.join(
            documents_folder,
            filename
        )

        if os.path.isfile(file_path):

            documents.append({
                "filename": filename,
                "size": os.path.getsize(file_path)
            })

    documents.sort(
        key=lambda item: item["filename"],
        reverse=True
    )

    return {
        "success": True,
        "documents": documents
    }

@app.post("/pdf-search")
def pdf_search(
    request: PDFSearchRequest
):

    if search_document is None:
        raise HTTPException(
            status_code=500,
            detail="Document search is not available."
        )

    if read_document is None:
        raise HTTPException(
            status_code=500,
            detail="Document reader is not available."
        )

    try:

        documents_folder = "documents"

        if not os.path.exists(documents_folder):
            raise HTTPException(
                status_code=404,
                detail="Documents folder was not found."
            )

        safe_filename = os.path.basename(
            request.filename
        )

        if safe_filename != request.filename:
            raise HTTPException(
                status_code=400,
                detail="Invalid document filename."
            )

        filename = os.path.join(
            documents_folder,
            safe_filename
        )

        if not os.path.isfile(filename):
            raise HTTPException(
                status_code=404,
                detail="Selected document was not found."
            )

        text = read_document(
            filename
        )

        results = search_document(
            text,
            request.query
        )

        return {
            "success": True,
            "query": request.query,
            "document": safe_filename,
            "results": results
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
# ============================================================
# PDF QUESTION
# ============================================================

@app.post("/pdf-question")
def pdf_question(
    request: PDFQuestionRequest
):

    if search_document is None:
        raise HTTPException(
            status_code=500,
            detail="Document search is not available."
        )

    if read_document is None:
        raise HTTPException(
            status_code=500,
            detail="Document reader is not available."
        )

    try:

        documents_folder = "documents"

        if not os.path.exists(documents_folder):
            raise HTTPException(
                status_code=404,
                detail="Documents folder was not found."
            )

        # Use the document selected by the user.
        safe_filename = os.path.basename(
            request.filename
        )

        # Prevent unsafe file paths.
        if safe_filename != request.filename:
            raise HTTPException(
                status_code=400,
                detail="Invalid document filename."
            )

        filename = os.path.join(
            documents_folder,
            safe_filename
        )

        # Check that the selected document exists.
        if not os.path.isfile(filename):
            raise HTTPException(
                status_code=404,
                detail="Selected document was not found."
            )

        # Read the selected document.
        text = read_document(
            filename
        )

        # Search only inside the selected document.
        results = search_document(
            text,
            request.question
        )

        if not results:
            return {
                "success": False,
                "question": request.question,
                "document": safe_filename,
                "answer": (
                    "I could not find relevant information "
                    "in the selected document."
                )
            }

        evidence_parts = []

        for result in results[:8]:

            if isinstance(result, dict):

                result_text = result.get(
                    "text",
                    result.get(
                        "content",
                        ""
                    )
                )

                if result_text:
                    evidence_parts.append(
                        result_text
                    )

            else:
                evidence_parts.append(
                    str(result)
                )

        document_evidence = (
            "\n\n".join(
                evidence_parts
            )
        )

        # Ask The Fix using information from the selected document.
        answer = ask_ollama(
            user_message=request.question,
            document_evidence=document_evidence
        )

        return {
            "success": True,
            "question": request.question,
            "document": safe_filename,
            "answer": answer,
            "results": results
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
# ============================================================
# DIRECT START
# ============================================================

if __name__ == "__main__":

    import uvicorn


    uvicorn.run(

        "main:app",

        host="127.0.0.1",

        port=8000,

        reload=False
    )