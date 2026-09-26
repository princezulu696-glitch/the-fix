import requests
from html.parser import HTMLParser
from urllib.parse import quote, unquote, urlparse, parse_qs
import re


# ============================================================
# SEARCH RESULT PARSER
# ============================================================

class SearchResultParser(HTMLParser):

    def __init__(self):
        super().__init__()

        self.results = []

        self.current_title = ""
        self.current_url = ""
        self.current_snippet = ""

        self.inside_result = False
        self.inside_link = False
        self.inside_snippet = False

    def handle_starttag(self, tag, attrs):

        attributes = dict(attrs)

        if tag == "a":

            class_name = attributes.get(
                "class",
                ""
            )

            href = attributes.get(
                "href",
                ""
            )

            if "result__a" in class_name:

                self.inside_result = True
                self.inside_link = True

                self.current_url = href
                self.current_title = ""
                self.current_snippet = ""

        if tag == "a":

            class_name = attributes.get(
                "class",
                ""
            )

            if "result__snippet" in class_name:

                self.inside_snippet = True

    def handle_data(self, data):

        cleaned = data.strip()

        if not cleaned:
            return

        if self.inside_result and self.inside_link:

            self.current_title += " " + cleaned

        elif self.inside_snippet:

            self.current_snippet += " " + cleaned

    def handle_endtag(self, tag):

        if tag == "a" and self.inside_link:

            self.inside_link = False

            self.results.append({
                "title": self.current_title.strip(),
                "url": self.current_url.strip(),
                "search_snippet": self.current_snippet.strip()
            })

            self.inside_result = False

        if tag == "a" and self.inside_snippet:

            self.inside_snippet = False


# ============================================================
# URL HANDLING
# ============================================================

def extract_real_url(url):

    if not url:
        return ""

    if url.startswith("//"):
        url = "https:" + url

    try:

        parsed = urlparse(url)

        query = parse_qs(
            parsed.query
        )

        if "uddg" in query:

            return unquote(
                query["uddg"][0]
            )

    except Exception:

        pass

    return url


# ============================================================
# DOMAIN
# ============================================================

def get_domain(url):

    try:

        domain = urlparse(url).netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    except Exception:

        return ""


# ============================================================
# SOURCE NAME
# ============================================================

def get_source_name(url):

    domain = get_domain(url)

    source_names = {

        "reuters.com": "Reuters",

        "bloomberg.com": "Bloomberg",

        "bbc.com": "BBC",

        "bbc.co.uk": "BBC",

        "apnews.com": "Associated Press",

        "theguardian.com": "The Guardian",

        "nytimes.com": "The New York Times",

        "washingtonpost.com": "The Washington Post",

        "cnn.com": "CNN",

        "techcrunch.com": "TechCrunch",

        "theverge.com": "The Verge",

        "arstechnica.com": "Ars Technica",

        "pcmag.com": "PCMag",

        "zdnet.com": "ZDNET",

        "aiweekly.co": "AI Weekly",

        "thedailyprompt.ai": "The Daily Prompt",

        "headsupai.io": "HeadsUp AI",

        "news.google.com": "Google News"
    }

    return source_names.get(
        domain,
        domain
    )


# ============================================================
# CLEAN HTML
# ============================================================

def clean_html(text):

    if not text:
        return ""

    text = re.sub(
        r"<script.*?</script>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    text = re.sub(
        r"<style.*?</style>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    text = re.sub(
        r"<noscript.*?</noscript>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"&nbsp;",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"&amp;",
        "&",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# FETCH PAGE
# ============================================================

def fetch_page_text(url):

    if not url:
        return ""

    try:

        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent":
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/153.0 Safari/537.36"
            }
        )

        if response.status_code != 200:
            return ""

        content_type = response.headers.get(
            "content-type",
            ""
        ).lower()

        if "text/html" not in content_type:
            return ""

        text = clean_html(
            response.text
        )

        if len(text) > 8000:
            text = text[:8000]

        return text

    except Exception:

        return ""


# ============================================================
# MAKE USEFUL SNIPPET
# ============================================================

def make_snippet(
    page_text,
    search_snippet,
    question
):

    candidates = []

    if page_text:
        candidates.append(page_text)

    if search_snippet:
        candidates.append(
            clean_html(search_snippet)
        )

    if not candidates:
        return ""

    text = max(
        candidates,
        key=len
    ).strip()

    if len(text) <= 900:
        return text

    question_words = [
        word.lower()
        for word in re.findall(
            r"[A-Za-z0-9]+",
            question
        )
        if len(word) > 3
    ]

    text_lower = text.lower()

    best_position = 0

    for word in question_words:

        position = text_lower.find(
            word
        )

        if position >= 0:

            best_position = position

            break

    start = max(
        0,
        best_position - 300
    )

    end = min(
        len(text),
        start + 900
    )

    return text[start:end].strip()


# ============================================================
# RESULT QUALITY
# ============================================================

def score_result(
    title,
    text,
    url,
    question
):

    score = 0

    title_lower = title.lower()
    text_lower = text.lower()

    question_words = [
        word.lower()
        for word in re.findall(
            r"[A-Za-z0-9]+",
            question
        )
        if len(word) > 3
    ]

    # --------------------------------------------------------
    # Text quality
    # --------------------------------------------------------

    if len(text) >= 200:
        score += 4

    elif len(text) >= 80:
        score += 2

    elif len(text) > 0:
        score += 1

    # --------------------------------------------------------
    # Question relevance
    # --------------------------------------------------------

    for word in question_words:

        if word in title_lower:
            score += 3

        elif word in text_lower:
            score += 1

    # --------------------------------------------------------
    # Useful AI/news domains
    # --------------------------------------------------------

    domain = get_domain(url)

    preferred_domains = [
        "reuters.com",
        "bbc.com",
        "bbc.co.uk",
        "apnews.com",
        "techcrunch.com",
        "theverge.com",
        "arstechnica.com",
        "pcmag.com",
        "aiweekly.co",
        "thedailyprompt.ai",
        "headsupai.io"
    ]

    if domain in preferred_domains:
        score += 3

    # --------------------------------------------------------
    # Penalize generic hub pages
    # --------------------------------------------------------

    path = urlparse(url).path.lower()

    generic_paths = [
        "/",
        "/news",
        "/latest",
        "/topics",
        "/category",
        "/categories",
        "/technology/artificial-intelligence/"
    ]

    if path in generic_paths:
        score -= 3

    # --------------------------------------------------------
    # Empty result penalty
    # --------------------------------------------------------

    if not text:
        score -= 10

    return score


# ============================================================
# WEB SEARCH
# ============================================================

def web_search(question: str):

    question = str(
        question
    ).strip()

    if not question:

        return {
            "success": False,
            "query": "",
            "answer": "",
            "source": "",
            "url": "",
            "results": []
        }

    try:

        encoded_question = quote(
            question
        )

        search_url = (
            "https://html.duckduckgo.com/html/"
            f"?q={encoded_question}"
        )

        response = requests.get(
            search_url,
            timeout=20,
            headers={
                "User-Agent":
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/153.0 Safari/537.36"
            }
        )

        response.raise_for_status()

        parser = SearchResultParser()

        parser.feed(
            response.text
        )

        raw_results = []

        for result in parser.results:

            title = result.get(
                "title",
                ""
            ).strip()

            duck_url = result.get(
                "url",
                ""
            ).strip()

            search_snippet = result.get(
                "search_snippet",
                ""
            ).strip()

            if not title:
                continue

            real_url = extract_real_url(
                duck_url
            )

            if not real_url:
                continue

            page_text = fetch_page_text(
                real_url
            )

            snippet = make_snippet(
                page_text,
                search_snippet,
                question
            )

            if not snippet:
                continue

            source_name = get_source_name(
                real_url
            )

            score = score_result(
                title,
                snippet,
                real_url,
                question
            )

            raw_results.append({
                "title": title,
                "text": snippet,
                "url": real_url,
                "source": source_name,
                "domain": get_domain(real_url),
                "score": score
            })

        # ----------------------------------------------------
        # Remove duplicate URLs
        # ----------------------------------------------------

        unique_results = []

        seen_urls = set()

        for result in raw_results:

            url = result["url"]

            if url in seen_urls:
                continue

            seen_urls.add(url)

            unique_results.append(
                result
            )

        # ----------------------------------------------------
        # Sort by quality
        # ----------------------------------------------------

        unique_results.sort(
            key=lambda item: item.get(
                "score",
                0
            ),
            reverse=True
        )

        # ----------------------------------------------------
        # Keep strongest results
        # ----------------------------------------------------

        results = unique_results[:5]

        return {
            "success": True,
            "query": question,
            "answer": "",
            "source": "DuckDuckGo",
            "url": search_url,
            "results": results
        }

    except requests.exceptions.Timeout:

        return {
            "success": False,
            "query": question,
            "answer": "",
            "source": "",
            "url": "",
            "results": [],
            "error":
                "Web search timed out."
        }

    except requests.exceptions.RequestException as error:

        return {
            "success": False,
            "query": question,
            "answer": "",
            "source": "",
            "url": "",
            "results": [],
            "error":
                f"Web search connection error: {error}"
        }

    except Exception as error:

        return {
            "success": False,
            "query": question,
            "answer": "",
            "source": "",
            "url": "",
            "results": [],
            "error":
                f"Web search error: {error}"
        }