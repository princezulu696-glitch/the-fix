import json
import os


MEMORY_FILE = "memory.json"


def load_memory():
    """Load saved memories from the memory file."""

    if not os.path.exists(MEMORY_FILE):
        return {}

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return {}


def save_memory(memory):
    """Save memories to the memory file."""

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, indent=4, ensure_ascii=False)


def remember(user_id, key, value):
    """Save a memory for a specific user."""

    memory = load_memory()

    if user_id not in memory:
        memory[user_id] = {}

    memory[user_id][key] = value

    save_memory(memory)


def get_memories(user_id):
    """Get all memories belonging to a specific user."""

    memory = load_memory()

    return memory.get(user_id, {})


def forget(user_id, key):
    """Delete one memory for a specific user."""

    memory = load_memory()

    if user_id in memory and key in memory[user_id]:
        del memory[user_id][key]
        save_memory(memory)
        return True

    return False