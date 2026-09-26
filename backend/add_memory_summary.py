from pathlib import Path

path = Path("memory.py")

text = path.read_text(encoding="utf-8")

marker = """
# --------------------------------
# FIND A MEMORY
# --------------------------------
"""

new_function = """
# --------------------------------
# MEMORY SUMMARY
# --------------------------------

def get_memory_summary(user_id):
    return {
        "profile": get_memories_by_category(
            user_id,
            "profile"
        ),
        "preferences": get_memories_by_category(
            user_id,
            "preference"
        ),
        "projects": get_memories_by_category(
            user_id,
            "project"
        ),
        "interests": get_memories_by_category(
            user_id,
            "interest"
        )
    }


"""

if "def get_memory_summary(" in text:
    print("Memory summary already exists.")
elif marker not in text:
    raise SystemExit(
        "Target section was not found. No changes made."
    )
else:
    text = text.replace(
        marker,
        new_function + marker,
        1
    )

    path.write_text(
        text,
        encoding="utf-8"
    )

    print(
        "Memory summary function added successfully."
    )