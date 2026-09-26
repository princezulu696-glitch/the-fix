import json
import os
from datetime import datetime


MEMORY_FILE = "memory.json"

MAX_CONVERSATION_HISTORY = 20


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


def get_user_memory(user_id):
    memory = load_memory()

    if user_id not in memory:
        memory[user_id] = {}

    return memory[user_id]


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

    user_memory[key] = value

    save_memory(memory)


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

    if key not in user_memory:
        return False

    del user_memory[key]

    save_memory(memory)

    return True


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
        user_memory["conversation_history"] = []

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
            "timestamp": datetime.now().isoformat()
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