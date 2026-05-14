from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

_ddg = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Web lookup for weather, news, sports, etc. Pass a concise English query."""
    return _ddg.invoke(query)
