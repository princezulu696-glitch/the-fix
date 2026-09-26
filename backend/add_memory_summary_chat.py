from pathlib import Path

path = Path("main.py")

text = path.read_text(encoding="utf-8")

old = """    # --------------------------------
    # PROJECT MEMORY RECALL
    # --------------------------------
"""

new = """    # --------------------------------
    # MEMORY SUMMARY RECALL
    # --------------------------------

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
                "Here is what I remember about you:\\n\\n"
                + "\\n".join(sections)
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

    # --------------------------------
    # PROJECT MEMORY RECALL
    # --------------------------------
"""

if "memory_summary_questions" in text:
    print("Memory summary chat feature already exists.")
elif old not in text:
    raise SystemExit(
        "Target section was not found. No changes made."
    )
else:
    text = text.replace(
        old,
        new,
        1
    )

    path.write_text(
        text,
        encoding="utf-8"
    )

    print(
        "Memory summary chat feature added successfully."
    )