from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(...)
    session_id: str = Field(default="default")

    @field_validator("message", mode="before")
    @classmethod
    def strip_message(cls, v: object) -> str:
        if not isinstance(v, str):
            raise TypeError("message must be a string")
        text = v.strip()
        if not text:
            raise ValueError("message must not be empty or whitespace-only")
        return text

    @field_validator("session_id", mode="before")
    @classmethod
    def strip_session(cls, v: object) -> str:
        if v is None:
            return "default"
        if isinstance(v, str) and not v.strip():
            return "default"
        if isinstance(v, str):
            return v.strip()
        return str(v)


class ChatResponse(BaseModel):
    response: str
    session_id: str


class DocumentUploadResponse(BaseModel):
    session_id: str
    filename: str
    chunks_indexed: int = Field(..., ge=1)
