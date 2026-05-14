from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage

from app.services.agent_registry import get_agent_graph


def dict_messages_to_lc(messages: list[dict[str, str]]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for row in messages:
        role, text = row["role"], row["content"]
        if role == "system":
            out.append(SystemMessage(content=text))
        elif role == "user":
            out.append(HumanMessage(content=text))
        elif role == "assistant":
            out.append(AIMessage(content=text))
    return out


def _text_delta(content: str | list | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                parts.append(str(block["text"]))
        return "".join(parts)
    return str(content)


def _chunk_requests_tools(chunk: AIMessageChunk) -> bool:
    if chunk.tool_calls or chunk.tool_call_chunks:
        return True
    if (chunk.additional_kwargs or {}).get("tool_calls"):
        return True
    finish = (chunk.response_metadata or {}).get("finish_reason")
    return finish in ("tool_calls", "tool_call")


async def _yield_text_in_slices(text: str, width: int = 14) -> AsyncGenerator[str, None]:
    if not text:
        return
    for i in range(0, len(text), width):
        yield text[i : i + width]


async def _yield_sliced(text: str) -> AsyncGenerator[str, None]:
    async for part in _yield_text_in_slices(text):
        yield part


async def stream_agent_answer(
    messages: Sequence[BaseMessage],
    *,
    include_web_search: bool = True,
) -> AsyncGenerator[str, None]:
    graph = get_agent_graph(include_web_search=include_web_search)
    run_id: str | None = None
    leg_parts: list[str] = []
    leg_wants_tools = False

    def _flush_leg() -> str | None:
        nonlocal leg_parts, leg_wants_tools
        if leg_wants_tools or not leg_parts:
            leg_parts = []
            leg_wants_tools = False
            return None
        text = "".join(leg_parts)
        leg_parts = []
        leg_wants_tools = False
        return text or None

    try:
        async for msg_chunk, meta in graph.astream(
            {"messages": list(messages)},
            stream_mode="messages",
        ):
            if meta.get("langgraph_node") != "model":
                continue

            if type(msg_chunk) is AIMessage and not isinstance(msg_chunk, AIMessageChunk):
                if msg_chunk.tool_calls:
                    continue
                piece = _text_delta(msg_chunk.content)
                if piece:
                    async for s in _yield_sliced(piece):
                        yield s
                continue

            if not isinstance(msg_chunk, AIMessageChunk):
                continue

            rid = msg_chunk.id
            if rid and run_id is not None and rid != run_id:
                emitted = _flush_leg()
                if emitted:
                    async for part in _yield_sliced(emitted):
                        yield part
            if rid:
                run_id = rid

            leg_wants_tools = leg_wants_tools or _chunk_requests_tools(msg_chunk)
            delta = _text_delta(msg_chunk.content)
            if delta:
                leg_parts.append(delta)

            if getattr(msg_chunk, "chunk_position", None) == "last":
                emitted = _flush_leg()
                if emitted:
                    async for part in _yield_sliced(emitted):
                        yield part
                run_id = None
    finally:
        emitted = _flush_leg()
        if emitted:
            async for part in _yield_sliced(emitted):
                yield part
