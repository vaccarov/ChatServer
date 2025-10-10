from typing import List, Optional
from pydantic import BaseModel


class Document(BaseModel):
	id: str
	filename: str
	content: str


class DocumentListResponse(BaseModel):
	documents: List[Document]


class SearchRequest(BaseModel):
	query: str
	embedding_model: str
	chat_id: Optional[str] = None


class RagChatRequest(BaseModel):
	query: str
	embedding_model: str
	chat_id: Optional[str] = None


class RagChatResponse(BaseModel):
	prompt: str
