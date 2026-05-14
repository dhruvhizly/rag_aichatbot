from __future__ import annotations

from langchain_core.tools import BaseTool

from app.tools.calculator_tool import calculator
from app.tools.search_tool import web_search


def build_tools(*, include_web_search: bool = True) -> list[BaseTool]:
    if include_web_search:
        return [web_search, calculator]
    return [calculator]
