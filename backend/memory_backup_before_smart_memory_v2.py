import json
import os
from datetime import datetime


MEMORY_FILE = "memory.json"

MAX_CONVERSATION_HISTORY = 20


# --------------------------------
# MEMORY CATEGORIES
# --------------------------------

MEMORY_CATEGORIES = {
    "name": "profile",
    "age": "profile",
    "location": "profile",
    "field of study": "profile",
    "occupation": "profile",

    "favorite color": "preference",
    "favorite colour": "preference",
    "favorite subject": "preference",
    "favorite programming language": "preference",
    "preference": "preference",
    "likes": "preference",
    "dislikes": "preference",

    "current project": "project",
    "project": "project",
    "project name": "project",

    "interest": "interest",
    "interests": "interest",
}


def load_memory():

    if not os.path.exists(MEMORY_FILE):
        return {}

    try:

        with open(
            MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(data, dict):
            return data

        return {}

    except (
        json.JSONDecodeError,
        OSError
    ):

        return {}


def save_memory(memory):

    with open(
        MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            memory,
            file,
            indent=4,
            ensure_ascii=False
        )


# --------------------------------
# KEY NORMALIZATION
# --------------------------------

def normalize_key(key):

    if not isinstance(key, str):
        return key

    key = key.strip().lower()

    aliases = {

        "favourite color":
            "favorite color",

        "favourite colour":
            "favorite color",

        "favorite colour":
            "favorite color",

        "favourite subject":
            "favorite subject",

        "favourite programming language":
            "favorite programming language",

        "field_of_study":
            "field of study",

        "current_project":
            "current project",

        "project_name":
            "project name",

        "programming language":
            "favorite programming language",
    }

    return aliases.get(
        key,
        key
    )


# --------------------------------
# GET USER MEMORY
# --------------------------------

def get_user_memory(user_id):

    memory = load_memory()

    if user_id not in memory:
        memory[user_id] = {}

    return memory[user_id]


# --------------------------------
# SAVE MEMORY
# --------------------------------

def remember(
    user_id,
    key,
    value
):

    memory = load_memory()

    if user_id not in memory:
        memory[user_id] = {}

    user_memory = memory[user_id]

    if not isinstance(
        user_memory,
        dict
    ):

        user_memory = {}

        memory[user_id] = user_memory

    key = normalize_key(key)

    if isinstance(value, str):
        value = value.strip()

    if not key or not value:
        return

    user_memory[key] = value

    save_memory(memory)


# --------------------------------
# GET ALL MEMORIES
# --------------------------------

def get_memories(user_id):

    memory = load_memory()

    user_memory = memory.get(
        user_id,
        {}
    )

    if not isinstance(
        user_memory,
        dict
    ):

        return {}

    return {
        key: value
        for key, value
        in user_memory.items()
        if key != "conversation_history"
    }


# --------------------------------
# GET MEMORIES BY CATEGORY
# --------------------------------

def get_memories_by_category(
    user_id,
    category
):

    memories = get_memories(
        user_id
    )

    results = {}

    for key, value in memories.items():

        normalized_key = normalize_key(
            key
        )

        memory_category = (
            MEMORY_CATEGORIES.get(
                normalized_key
            )
        )

        if memory_category == category:

            results[
                normalized_key
            ] = value

    return results


# --------------------------------
# FIND A MEMORY
# --------------------------------

def get_memory(
    user_id,
    key
):

    memories = get_memories(
        user_id
    )

    key = normalize_key(key)

    return memories.get(key)


# --------------------------------
# CHECK IF MEMORY EXISTS
# --------------------------------

def memory_exists(
    user_id,
    key
):

    return (
        get_memory(
            user_id,
            key
        ) is not None
    )


# --------------------------------
# FORGET MEMORY
# --------------------------------

def forget(
    user_id,
    key
):

    memory = load_memory()

    if user_id not in memory:
        return False

    user_memory = memory[user_id]

    if not isinstance(
        user_memory,
        dict
    ):

        return False

    key = normalize_key(key)

    if key not in user_memory:
        return False

    del user_memory[key]

    save_memory(memory)

    return True


# --------------------------------
# CONVERSATION MEMORY
# --------------------------------

def add_conversation(
    user_id,
    user_message,
    assistant_message
):

    memory = load_memory()

    if user_id not in memory:
        memory[user_id] = {}

    user_memory = memory[user_id]

    if not isinstance(
        user_memory,
        dict
    ):

        user_memory = {}

        memory[user_id] = user_memory

    if "conversation_history" not in user_memory:

        user_memory[
            "conversation_history"
        ] = []

    history = user_memory[
        "conversation_history"
    ]

    if not isinstance(
        history,
        list
    ):

        history = []

    history.append(
        {
            "user": user_message,
            "assistant": assistant_message,
            "timestamp":
                datetime.now().isoformat()
        }
    )

    if len(history) > MAX_CONVERSATION_HISTORY:

        history = history[
            -MAX_CONVERSATION_HISTORY:
        ]

    user_memory[
        "conversation_history"
    ] = history

    save_memory(memory)


def get_conversation_history(
    user_id
):

    memory = load_memory()

    if user_id not in memory:
        return []

    user_memory = memory[user_id]

    if not isinstance(
        user_memory,
        dict
    ):

        return []

    history = user_memory.get(
        "conversation_history",
        []
    )

    if not isinstance(
        history,
        list
    ):

        return []

    return history[
        -MAX_CONVERSATION_HISTORY:
    ]


def clear_conversation_history(
    user_id
):

    memory = load_memory()

    if user_id not in memory:
        return False

    user_memory = memory[user_id]

    if not isinstance(
        user_memory,
        dict
    ):

        return False

    user_memory[
        "conversation_history"
    ] = []

    save_memory(memory)

    return True