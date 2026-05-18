import asyncio
from typing import AsyncGenerator

from app.services.llm_service import SYSTEM_PROMPT, LLMService
from app.services.rag_registry import get_rag_service


class ChatService:
    def __init__(self) -> None:
        self._histories: dict[str, list[dict]] = {}
        self._llm = LLMService()

    def _get_history(self, session_id: str) -> list[dict]:
        if session_id not in self._histories:
            self._histories[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        return self._histories[session_id]

    def clear_history(self, session_id: str) -> None:
        self._histories.pop(session_id, None)

    async def stream_response(
        self,
        user_message: str,
        session_id: str,
    ) -> AsyncGenerator[str, None]:
        history = self._get_history(session_id)
        rag = get_rag_service()
        retrieval, doc_session = await asyncio.to_thread(
            rag.retrieval_and_presence, session_id, user_message
        )

        user_for_llm = self._user_turn_with_retrieval(
            user_message,
            retrieval,
            doc_session=doc_session,
        )
        messages = [*history, {"role": "user", "content": user_for_llm}]
        full_reply: list[str] = []

        async for token in self._llm.stream_chat(messages):
            full_reply.append(token)
            
            yield token

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "".join(full_reply)})

    @staticmethod
    def _user_turn_with_retrieval(
        user_message: str,
        retrieval_block: str,
        *,
        doc_session: bool,
    ) -> str:
        if not doc_session:
            policy = (
                "\n\n[No documents have been uploaded for this session. "
                "Reply in one or two sentences that you can only answer questions about uploaded documents, "
                "and ask the user to upload a PDF or TXT file first. Do not answer from general knowledge.]\n"
            )
        else:
            policy = (
                "\n\n[Document-only mode: answer strictly from the retrieved excerpts above. "
                "If the excerpts do not contain the answer, reply in one or two sentences that the information "
                "is not in the uploaded documents. Do not answer from general knowledge, do not speculate, "
                "and do not invent details.]\n"
            )
        return (
            "Retrieved document excerpts for this session (may be empty or irrelevant):\n\n"
            f"{retrieval_block}\n"
            f"{policy}\n"
            "---\n"
            f"User message:\n{user_message}"
        )
