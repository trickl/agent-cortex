"""
arxiv_multi_tools.py
====================
Four arXiv search helpers ready for function-calling agents.

Tools exposed
-------------
1. arxiv_search_topic(query, max_results)
2. arxiv_search_author(author, query="", max_results)
3. arxiv_search_year(year, query="", max_results)
4. arxiv_search_month(year, month, query="", max_results)

All of them:
• Call ONLY https://export.arxiv.org
• Return a list[dict] → easy post-processing
• Depend only on `requests` + `pydantic`
"""

from __future__ import annotations

import calendar
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict

import requests
from pydantic import BaseModel, Field, ValidationError, conint

try:
    from .tool_decorator import register_tool
except ImportError:
    # If tool_decorator doesn't exist, create a dummy decorator
    def register_tool(tags=None):
        def decorator(func):
            return func
        return decorator

# ---------------------------------------------------------------------------#
# Low-level fetch                                                             #
# ---------------------------------------------------------------------------#


def _fetch_arxiv_feed(query: str, max_results: int) -> ET.Element:
    encoded = urllib.parse.quote_plus(query)
    url = (
        "https://export.arxiv.org/api/query?search_query="
        + encoded
        + f"&start=0&max_results={max_results}"
        + "&sortBy=submittedDate&sortOrder=descending"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return ET.fromstring(resp.text)


_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_entries(root: ET.Element) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for entry in root.findall("atom:entry", _NS):
        out.append(
            {
                "title": (entry.find("atom:title", _NS).text or "").strip().replace("\n", " "),
                "url": entry.find("atom:link[@rel='alternate']", _NS).attrib["href"],
                "summary": (entry.find("atom:summary", _NS).text or "").strip().replace("\n", " "),
                "published": (entry.find("atom:published", _NS).text or "")[:10],
                "authors": [a.text for a in entry.findall("atom:author/atom:name", _NS)],
            }
        )
    return out


# ---------------------------------------------------------------------------#
# Public generic search                                                      #
# ---------------------------------------------------------------------------#


def _generic_search(query: str, pull: int = 50) -> List[Dict[str, str]]:
    """Fetch *pull* newest papers for *query* (no local filtering)."""
    root = _fetch_arxiv_feed(query=f"all:{query}", max_results=pull)
    return _parse_entries(root)


# ---------------------------------------------------------------------------#
# Tool 1: Topic search                                                       #
# ---------------------------------------------------------------------------#


def arxiv_search_topic(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """Search by general topic or keywords."""
    return _generic_search(query, max_results)[:max_results]


# ---------------------------------------------------------------------------#
# Tool 2: Author search                                                      #
# ---------------------------------------------------------------------------#


def arxiv_search_author(
    author: str, query: str | None = "", max_results: int = 10
) -> List[Dict[str, str]]:
    """
    Search papers by a specific author. Optional *query* narrows the topic.
    ArXiv author syntax: `au:"Smith_J"`, but plain text usually works.
    """
    q = f'au:"{author}"'
    if query:
        q += f" AND all:{query}"
    return _generic_search(q, max_results)[:max_results]


# ---------------------------------------------------------------------------#
# Tool 3: Year search                                                        #
# ---------------------------------------------------------------------------#


def arxiv_search_year(
    year: int, query: str | None = "", max_results: int = 10
) -> List[Dict[str, str]]:
    """Return papers published in *year* (YYYY)."""
    pull = max_results * 3  # grab more, then filter
    res = _generic_search(query, pull)
    filtered = [p for p in res if p["published"].startswith(str(year))]
    return filtered[:max_results]


# ---------------------------------------------------------------------------#
# Tool 4: Month search                                                       #
# ---------------------------------------------------------------------------#


def arxiv_search_month(
    year: int, month: int, query: str | None = "", max_results: int = 10
) -> List[Dict[str, str]]:
    """Return papers from a given month (1-12 of *year*)."""
    if not (1 <= month <= 12):
        raise ValueError("month must be 1-12")
    pull = max_results * 4
    res = _generic_search(query, pull)
    ym_prefix = f"{year:04d}-{month:02d}"
    filtered = [p for p in res if p["published"].startswith(ym_prefix)]
    return filtered[:max_results]


# ---------------------------------------------------------------------------#
# Optional wrappers for JSON-string inputs (function-calling)                #
# ---------------------------------------------------------------------------#


class _BaseInput(BaseModel):
    max_results: conint(gt=0, le=50) = 10


class TopicInput(_BaseInput):
    query: str


class AuthorInput(_BaseInput):
    author: str
    query: str | None = ""


class YearInput(_BaseInput):
    year: conint(ge=1991)  # arXiv start
    query: str | None = ""


class MonthInput(_BaseInput):
    year: conint(ge=1991)
    month: conint(ge=1, le=12)
    query: str | None = ""


def _tool_wrapper(input_cls, fn):
    def _inner(raw: str) -> str:  # JSON → markdown
        try:
            args = input_cls.parse_raw(raw)
        except ValidationError as e:
            return f"⚠️ Invalid arguments: {e}"
        papers = fn(**args.dict())
        if not papers:
            return "No matching papers found."
        lines = []
        for i, p in enumerate(papers, 1):
            authors = ", ".join(p["authors"])[:120]
            lines.append(
                f"{i}. **{p['title']}** ({p['published']})  \\\n"
                f"   {authors}  \\\n"
                f"   {p['url']}"
            )
        return "\n\n".join(lines)

    return _inner


# Register tools with the tool registry system
@register_tool(tags=["arxiv", "research", "papers", "search", "topic"])
def arxiv_search_topic_tool(raw: str) -> str:
    """Search arXiv by general topic / keywords
    
    Args:
        raw: JSON string containing query and max_results parameters
        
    Returns:
        Markdown formatted list of papers matching the search criteria
    """
    return _tool_wrapper(TopicInput, arxiv_search_topic)(raw)

@register_tool(tags=["arxiv", "research", "papers", "search", "author"])
def arxiv_search_author_tool(raw: str) -> str:
    """Search arXiv for papers by an author, optionally filtered by topic
    
    Args:
        raw: JSON string containing author, query (optional) and max_results parameters
        
    Returns:
        Markdown formatted list of papers matching the search criteria
    """
    return _tool_wrapper(AuthorInput, arxiv_search_author)(raw)

@register_tool(tags=["arxiv", "research", "papers", "search", "year"])
def arxiv_search_year_tool(raw: str) -> str:
    """Search arXiv for papers published in a specific year
    
    Args:
        raw: JSON string containing year, query (optional) and max_results parameters
        
    Returns:
        Markdown formatted list of papers matching the search criteria
    """
    return _tool_wrapper(YearInput, arxiv_search_year)(raw)

@register_tool(tags=["arxiv", "research", "papers", "search", "month"])
def arxiv_search_month_tool(raw: str) -> str:
    """Search arXiv for papers from a specific month (YYYY-MM)
    
    Args:
        raw: JSON string containing year, month, query (optional) and max_results parameters
        
    Returns:
        Markdown formatted list of papers matching the search criteria
    """
    return _tool_wrapper(MonthInput, arxiv_search_month)(raw)

# For compatibility with external function calling APIs
functions = [
    {
        "name": "arxiv_search_topic_tool",
        "description": "Search arXiv by general topic / keywords",
        "parameters": TopicInput.schema(),
    },
    {
        "name": "arxiv_search_author_tool",
        "description": "Search arXiv for papers by an author, optionally filtered by topic",
        "parameters": AuthorInput.schema(),
    },
    {
        "name": "arxiv_search_year_tool",
        "description": "Search arXiv for papers published in a specific year",
        "parameters": YearInput.schema(),
    },
    {
        "name": "arxiv_search_month_tool",
        "description": "Search arXiv for papers from a specific month (YYYY-MM)",
        "parameters": MonthInput.schema(),
    },
]


# ---------------------------------------------------------------------------#
# Quick manual test                                                          #
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    print("Topic search:")
    for p in arxiv_search_topic("quantum error correction", 3):
        print("-", p["title"])

    print("\nAuthor + topic:")
    for p in arxiv_search_author("Yann LeCun", "energy based", 2):
        print("-", p["title"])

    print("\nYear filter 2024:")
    for p in arxiv_search_year(2024, "LLM", 2):
        print("-", p["title"], p["published"])

    print("\nMonth filter 2025-03:")
    for p in arxiv_search_month(2025, 3, "reinforcement", 2):
        print("-", p["title"], p["published"])
