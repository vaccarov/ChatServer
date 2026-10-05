from pydantic import BaseModel

from app.core.constants import DEFAULT_LLM_BASE_URL


class Document(BaseModel):
	id: str
	filename: str
	content: str


class RagChatRequest(BaseModel):
	query: str
	embedding_model: str
	chat_id: str | None = None
	# Which server the ChatServer must call to turn text into vectors.
	embedding_provider: str = 'ollama'
	embedding_base_url: str = DEFAULT_LLM_BASE_URL


class RagChatResponse(BaseModel):
	prompt: str
