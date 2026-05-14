from __future__ import annotations

from functools import lru_cache

from langchain.agents import create_agent
from langchain_groq import ChatGroq

from app.config import get_settings
from app.tools.toolkit import build_tools


@lru_cache(maxsize=2)
def get_agent_graph(include_web_search: bool = True):
    settings = get_settings()
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.model,
        temperature=settings.agent_temperature,
    )
    return create_agent(
        llm,
        tools=build_tools(include_web_search=include_web_search),
        system_prompt=None,
    )
