import re


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(text, chunk_size=1200, overlap=200):
    text = clean_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            boundary = text.rfind(".", start, end)

            if boundary > start + 400:
                end = boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def keyword_score(query, chunk):
    query_words = set(
        re.findall(r"[a-zA-Z0-9]+", query.lower())
    )

    chunk_words = set(
        re.findall(r"[a-zA-Z0-9]+", chunk.lower())
    )

    important_words = {
        word for word in query_words
        if len(word) > 2
    }

    return len(important_words.intersection(chunk_words))


def search_document(text, query, max_results=5):
    chunks = split_text(text)

    if not chunks:
        return []

    scored_chunks = []

    for chunk in chunks:
        score = keyword_score(query, chunk)

        if score > 0:
            scored_chunks.append(
                (score, chunk)
            )

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        {
            "score": score,
            "text": chunk
        }
        for score, chunk in scored_chunks[:max_results]
    ]