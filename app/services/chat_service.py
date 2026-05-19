import asyncio
import logging
import random
import re
import time
from typing import AsyncGenerator

from app.config import get_settings
from app.services.llm_service import SYSTEM_PROMPT, LLMService
from app.services.rag_registry import get_rag_service

logger = logging.getLogger(__name__)

# Shown while the request is being processed (retrieval + LLM, until first token).
# Add/remove freely — one is picked at random per request, then cycled every
# STATUS_CYCLE_INTERVAL_SECS while still waiting.
GENERATING_STATUSES = [
    "Please wait while I fetch the details for you…",
    "Looking through your documents…",
    "Generating answer…",
    "Putting together a response…",
    "Composing the answer…",
    "Drafting the reply…",
    "Almost there…",
    "Fetching details…",
    "Searching through documents…",
    "Looking up relevant excerpts…",
    "Checking documents…",
    "Finding the right section…",
    "Digging into detail...",
    "Just a minute...",
]

# How often to rotate to a new status while a phase is still running.
STATUS_CYCLE_INTERVAL_SECS = 5.0

_PRONOUN_RE = re.compile(
    r"\b(it|its|that|those|this|these|they|them|their|he|she|him|her|same)\b",
    re.IGNORECASE,
)

_CHITCHAT_RE = re.compile(
    r"^\s*("
    # greetings
    r"hi+|hello+|hey+|yo+|hiya|howdy|sup|greetings|"
    r"good\s*(morning|afternoon|evening|night)|"
    r"morning|afternoon|evening|"
    # thanks / acknowledgements
    r"thanks?(\s+(a\s+lot|so\s+much|much|again))?|thank\s+you(\s+(so\s+much|very\s+much))?|"
    r"ty|thx|cheers|appreciated|much\s+appreciated|"
    # affirmations / fillers
    r"ok(ay)?|k|kk|cool|nice|great|awesome|perfect|excellent|"
    r"sure|fine|alright|right|sounds\s+good|makes\s+sense|fair\s+enough|"
    r"got\s+it|understood|noted|gotcha|i\s+see|"
    r"yes|yep|yeah|yup|yh|"
    r"no|nope|nah|"
    # farewells
    r"bye+|goodbye|see\s+ya|see\s+you|cya|later|talk\s+(to\s+you\s+)?later|ttyl|peace|"
    # apologies / politeness
    r"sorry|my\s+bad|no\s+(problem|worries)|np|you('|\s+a)re\s+welcome|welcome|please|"
    # meta / identity / capability
    r"who\s+are\s+you|what\s+(are\s+you|do\s+you\s+do|can\s+you\s+do)|"
    r"help|help\s+me|what\s+can\s+you\s+help\s+with|how\s+do\s+you\s+work|"
    r"are\s+you\s+(there|alive|working|online)|you\s+there|"
    # small talk
    r"how\s+are\s+you(\s+doing)?|how('|\s+i)s\s+it\s+going|how('|\s+i)s\s+everything|"
    r"what('|\s+i)s\s+up|whats\s+up|wassup|"
    # conversation enders / fillers
    r"that('|\s+i)s\s+all|no\s+more\s+questions|i('|\s+a)m\s+done|done|"
    r"nothing|never\s*mind|nvm|forget\s+it|skip"
    r")[\s!.?,]*$",
    re.IGNORECASE,
)


def _looks_like_followup(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 80:
        return False
    return bool(_PRONOUN_RE.search(t))


def _is_chitchat(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 40:
        return False
    return bool(_CHITCHAT_RE.match(t))


class ChatService:
    def __init__(self) -> None:
        settings = get_settings()
        self._histories: dict[str, list[dict]] = {}
        self._last_retrieval: dict[str, tuple[str, bool]] = {}
        self._llm = LLMService()
        self._history_max_turns = settings.history_max_turns

    def _get_history(self, session_id: str) -> list[dict]:
        if session_id not in self._histories:
            self._histories[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        return self._histories[session_id]

    def clear_history(self, session_id: str) -> None:
        self._histories.pop(session_id, None)
        self._last_retrieval.pop(session_id, None)

    async def stream_response(
        self,
        user_message: str,
        session_id: str,
    ) -> AsyncGenerator[tuple[str, str], None]:
        history = self._get_history(session_id)
        rag = get_rag_service()

        prior = self._last_retrieval.get(session_id)
        skip_retrieval = _is_chitchat(user_message)
        reuse_prior = (
            not skip_retrieval
            and prior is not None
            and _looks_like_followup(user_message)
        )

        # Simple paths (no cycling): chitchat skips RAG, follow-up reuses prior excerpts.
        if skip_retrieval or reuse_prior:
            if skip_retrieval:
                retrieval, doc_session = "", False
                user_for_llm = user_message
            else:
                retrieval, doc_session = prior  # type: ignore[misc]
                user_for_llm = self._user_turn_with_retrieval(
                    user_message, retrieval, doc_session=doc_session
                )
                yield ("status", random.choice(GENERATING_STATUSES))

            messages = [*history, {"role": "user", "content": user_for_llm}]
            full_reply: list[str] = []
            async for token in self._llm.stream_chat(messages):
                full_reply.append(token)
                yield ("token", token)

            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": "".join(full_reply)})
            self._trim_history(history)
            return

        # Full RAG path with cycling status messages.
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        async def cycle(messages: list[str], stop: asyncio.Event) -> None:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(
                        stop.wait(), timeout=STATUS_CYCLE_INTERVAL_SECS
                    )
                    return
                except asyncio.TimeoutError:
                    await queue.put(("status", random.choice(messages)))

        async def worker() -> None:
            full_reply: list[str] = []
            t0 = time.perf_counter()
            try:
                # Single status phase: cycle from retrieval start until first token.
                await queue.put(("status", random.choice(GENERATING_STATUSES)))
                stop = asyncio.Event()
                cycler = asyncio.create_task(cycle(GENERATING_STATUSES, stop))
                try:
                    retrieval, doc_session = await asyncio.to_thread(
                        rag.retrieval_and_presence, session_id, user_message
                    )
                    self._last_retrieval[session_id] = (retrieval, doc_session)
                    t_retrieval = time.perf_counter() - t0

                    user_for_llm = self._user_turn_with_retrieval(
                        user_message, retrieval, doc_session=doc_session
                    )
                    messages = [*history, {"role": "user", "content": user_for_llm}]

                    t1 = time.perf_counter()
                    first_token_at: float | None = None
                    async for token in self._llm.stream_chat(messages):
                        if first_token_at is None:
                            first_token_at = time.perf_counter() - t1
                            stop.set()
                        full_reply.append(token)
                        await queue.put(("token", token))
                finally:
                    stop.set()
                    await cycler

                t_total = time.perf_counter() - t1
                logger.info(
                    "chat timing: retrieval=%.2fs ttft=%.2fs llm_total=%.2fs",
                    t_retrieval,
                    first_token_at if first_token_at is not None else -1.0,
                    t_total,
                )

                history.append({"role": "user", "content": user_message})
                history.append(
                    {"role": "assistant", "content": "".join(full_reply)}
                )
                self._trim_history(history)
            except Exception as exc:  # surface to consumer via sentinel
                await queue.put(("__error__", repr(exc)))
            finally:
                await queue.put(SENTINEL)

        worker_task = asyncio.create_task(worker())
        try:
            while True:
                item = await queue.get()
                if item is SENTINEL:
                    break
                kind, payload = item
                if kind == "__error__":
                    raise RuntimeError(payload)
                yield (kind, payload)
        finally:
            await worker_task

    def _trim_history(self, history: list[dict]) -> None:
        max_messages = 1 + 2 * self._history_max_turns
        if len(history) <= max_messages:
            return
        keep_tail = max_messages - 1
        del history[1 : len(history) - keep_tail]

    @staticmethod
    def _user_turn_with_retrieval(
        user_message: str,
        retrieval_block: str,
        *,
        doc_session: bool,
    ) -> str:
        return (
            f"[Retrieved excerpts]\n{retrieval_block}\n\n"
            f"[User]\n{user_message}"
        )
