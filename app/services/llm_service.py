from pathlib import Path
from typing import AsyncGenerator

from ollama import AsyncClient

from app.config import get_settings

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt"
SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8").strip()


class LLMService:
    def __init__(self) -> None:
        settings = get_settings()
        self.client = AsyncClient(host=settings.ollama_base_url)
        self.model = settings.model

    async def stream_chat(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        stream = await self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={"temperature": 0.7},
        )
        async for chunk in stream:
            content = chunk.get("message", {}).get("content") if isinstance(chunk, dict) else getattr(chunk.message, "content", None)
            if content:
                yield content

    async def chat(self, messages: list[dict]) -> str:
        response = await self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            options={"temperature": 0.7},
        )
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "") or ""
        return getattr(response.message, "content", "") or ""
