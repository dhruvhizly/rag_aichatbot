from typing import AsyncGenerator

from langchain_core.messages import HumanMessage

from app.services.agent_service import dict_messages_to_lc, stream_agent_answer
from app.services.llm_service import SYSTEM_PROMPT
from app.services.rag_registry import get_rag_service


class ChatService:
    def __init__(self) -> None:
        self._histories: dict[str, list[dict]] = {}

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
        retrieval, anchor_to_uploads = rag.retrieval_for_chat(session_id, user_message)
        doc_session = rag.session_has_indexed_documents(session_id)
        include_web_search = not anchor_to_uploads

        user_for_llm = self._user_turn_with_retrieval(
            user_message,
            retrieval,
            doc_session=doc_session,
            anchor_to_uploads=anchor_to_uploads,
        )
        agent_messages = [*dict_messages_to_lc(history), HumanMessage(content=user_for_llm)]
        full_reply: list[str] = []

        async for token in stream_agent_answer(
            agent_messages,
            include_web_search=include_web_search,
        ):
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
        anchor_to_uploads: bool,
    ) -> str:
        policy = ""
        if anchor_to_uploads:
            policy = (
                "\n\n[Strict document mode: retrieved excerpts match this question well. "
                "Answer document facts only from the excerpts. "
                "If they do not contain the answer, reply in one or two sentences that it is not in the uploads "
                "— do not add unrelated bullets from the file. Web search is disabled for this turn.]\n"
            )
        elif doc_session:
            policy = (
                "\n\n[General / live question mode: uploads exist but are not closely matched to this query. "
                "Use web search for weather, news, sports, or other external facts. "
                "Use the calculator only for explicit math. Do not pretend upload excerpts answered a "
                "question they do not relate to.]\n"
            )
        return (
            "Retrieved document excerpts for this session (may be empty or irrelevant):\n\n"
            f"{retrieval_block}\n"
            f"{policy}\n"
            "---\n"
            f"User message:\n{user_message}"
        )
