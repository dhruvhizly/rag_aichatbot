from pathlib import Path
from typing import AsyncGenerator

from ollama import AsyncClient

from app.config import get_settings

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
SYSTEM_PROMPT = (_PROMPT_DIR / "system_prompt.txt").read_text(encoding="utf-8").strip()
CLASSIFIER_TEMPLATE = (_PROMPT_DIR / "relevance_classifier.txt").read_text(encoding="utf-8").strip()


class LLMService:
    def __init__(self) -> None:
        settings = get_settings()
        self.client = AsyncClient(host=settings.ollama_base_url)
        self.model = settings.model
        self.keep_alive = settings.llm_keep_alive
        self._options = {
            "temperature": settings.agent_temperature,
            "num_ctx": settings.llm_num_ctx,
            "num_predict": settings.llm_num_predict,
        }

    async def stream_chat(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        stream = await self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            keep_alive=self.keep_alive,
            options=self._options,
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
            keep_alive=self.keep_alive,
            options=self._options,
        )
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "") or ""
        return getattr(response.message, "content", "") or ""

    async def prewarm(self) -> None:
        await self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "ok"},
            ],
            stream=False,
            keep_alive=self.keep_alive,
            options={**self._options, "num_predict": 1},
        )
