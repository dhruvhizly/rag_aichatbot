import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models import ChatRequest
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])
chat_service = ChatService()


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:

        async def event_generator():
            try:
                async for token in chat_service.stream_response(
                    user_message=request.message,
                    session_id=request.session_id,
                ):
                    payload = token.replace("\r\n", "\n").replace("\n", "\ndata: ")
                    yield f"data: {payload}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.exception("Error while streaming chat response")
                err = str(exc).replace("\n", " ")
                yield f"event: error\ndata: {err}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as exc:
        logger.exception("Error in chat endpoint")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    chat_service.clear_history(session_id)
    return {"message": f"Session '{session_id}' cleared"}
