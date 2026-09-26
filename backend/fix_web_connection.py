from pathlib import Path


MAIN_FILE = Path("main.py")


NEW_WEB_SEARCH = r'''
    # ========================================================
    # WEB SEARCH
    # ========================================================

    web_search_result = None

    web_search_triggers = [
        "latest",
        "today",
        "current",
        "right now",
        "recent",
        "news",
        "weather",
        "price",
        "prices",
        "stock",
        "stocks",
        "score",
        "scores",
        "this week",
        "this month",
        "what happened",
        "who is the current",
        "who is currently",
        "current information"
    ]

    message_lower = message.lower()

    should_search_web = any(
        trigger in message_lower
        for trigger in web_search_triggers
    )

    if should_search_web:

        try:

            web_search_result = web_search(
                message
            )

        except Exception as error:

            web_search_result = {
                "success": False,
                "query": message,
                "answer": "",
                "source": "",
                "url": "",
                "results": [],
                "error": str(error)
            }
'''


def main():

    if not MAIN_FILE.exists():

        print("ERROR: main.py was not found.")

        return

    text = MAIN_FILE.read_text(
        encoding="utf-8"
    )

    web_start_marker = (
        "    # ========================================================\n"
        "    # WEB SEARCH\n"
        "    # ========================================================\n"
    )

    local_ai_marker = (
        "    # ========================================================\n"
        "    # LOCAL AI\n"
        "    # ========================================================\n"
    )

    web_start = text.find(
        web_start_marker
    )

    if web_start == -1:

        print(
            "ERROR: WEB SEARCH section was not found."
        )

        return

    local_ai_start = text.find(
        local_ai_marker,
        web_start
    )

    if local_ai_start == -1:

        print(
            "ERROR: LOCAL AI section was not found."
        )

        return

    backup_file = Path(
        "main_before_web_fix.py"
    )

    backup_file.write_text(
        text,
        encoding="utf-8"
    )

    new_text = (
        text[:web_start]
        + NEW_WEB_SEARCH
        + "\n\n"
        + text[local_ai_start:]
    )

    MAIN_FILE.write_text(
        new_text,
        encoding="utf-8"
    )

    print(
        "Web search section replaced successfully."
    )

    print(
        "Backup created: main_before_web_fix.py"
    )


if __name__ == "__main__":
    main()