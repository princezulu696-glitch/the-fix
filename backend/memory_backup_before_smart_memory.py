import json
import os


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
            return json.load(file)

    except (json.JSONDecodeError, OSError):
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

    memory[user_id][key] = value

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
        for key, value in user_memory.items()
        if key != "conversation_history"
    }


def forget(
    user_id,
    key
):

    memory = load_memory()

    if (
        user_id in memory
        and key in memory[user_id]
    ):

        del memory[user_id][key]

        save_memory(memory)

        return True

    return False


def add_conversation(
    user_id,
    user_message,
    assistant_message
):

    memory = load_memory()

    if user_id not in memory:
        memory[user_id] = {}

    if "conversation_history" not in memory[user_id]:
        memory[user_id]["conversation_history"] = []

    history = memory[user_id][
        "conversation_history"
    ]

    history.append(
        {
            "user": user_message,
            "assistant": assistant_message
        }
    )

    if len(history) > MAX_CONVERSATION_HISTORY:

        history = history[
            -MAX_CONVERSATION_HISTORY:
        ]

        memory[user_id][
            "conversation_history"
        ] = history

    save_memory(memory)


def get_conversation_history(
    user_id
):

    memory = load_memory()

    if user_id not in memory:
        return []

    history = memory[user_id].get(
        "conversation_history",
        []
    )

    if not isinstance(
        history,
        list
    ):
        return []

    return history[-MAX_CONVERSATION_HISTORY:]


def clear_conversation_history(
    user_id
):

    memory = load_memory()

    if user_id not in memory:
        return False

    memory[user_id][
        "conversation_history"
    ] = []

    save_memory(memory)

    return True