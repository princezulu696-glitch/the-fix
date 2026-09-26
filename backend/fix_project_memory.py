from pathlib import Path

path = Path("main.py")

text = path.read_text(encoding="utf-8")

old = """    # --------------------------------
    # SAVE MEMORY
    # --------------------------------
"""

new = """    # --------------------------------
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
"""

if old not in text:
    raise SystemExit("Target section was not found. No changes made.")

path.write_text(
    text.replace(old, new, 1),
    encoding="utf-8"
)

print("Project memory recall added successfully.")