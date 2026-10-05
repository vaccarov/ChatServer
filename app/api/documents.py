from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.constants import DEFAULT_LLM_BASE_URL
from app.schemas.document import RagChatRequest, RagChatResponse
from app.services import document_service

router = APIRouter()


@router.post('/upload', summary='Upload documents for RAG')
async def upload_documents(
	files: Annotated[list[UploadFile], File()],
	embedding_model: Annotated[str, Form()],
	chat_id: Annotated[str | None, Form()] = None,
	embedding_provider: Annotated[str, Form()] = 'ollama',
	embedding_base_url: Annotated[str, Form()] = DEFAULT_LLM_BASE_URL,
) -> dict[str, Any]:
	"""
	Uploads one or more documents and adds them to a vector collection
	specific to the chosen embedding model.
	"""
	try:
		document_service.add_documents_to_collection(
			files=files,
			collection_name=embedding_model,
			chat_id=chat_id,
			provider=embedding_provider,
			base_url=embedding_base_url,
		)
		return {'message': f"Successfully uploaded {len(files)} file(s) to collection '{embedding_model}'."}
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f'Failed to upload documents: {e}')


@router.get('/list', summary='List documents in a RAG collection for a chat')
async def list_documents_in_collection(
	embedding_model: str | None = None, chat_id: str | None = None
) -> list[dict[str, Any]]:
	"""
	Lists all unique documents. Can be filtered by collection (embedding_model)
	and/or chat_id.
	"""
	return document_service.list_documents(collection_name=embedding_model, chat_id=chat_id)


@router.post('/rag_chat', response_model=RagChatResponse, summary='Get augmented prompt for RAG chat')
def rag_chat(request: RagChatRequest) -> RagChatResponse:
	"""
	Performs a RAG search to find relevant documents and returns an augmented prompt.
	"""
	relevant_docs = document_service.get_relevant_documents(
		query=request.query,
		collection_name=request.embedding_model,
		chat_id=request.chat_id,
		provider=request.embedding_provider,
		base_url=request.embedding_base_url,
	)
	if not relevant_docs:
		return RagChatResponse(prompt=request.query)
	context = '\n\n'.join(doc.content for doc in relevant_docs)
	augmented_prompt = (
		f'Using the following context, please answer the question.\n\n'
		f'---\n'
		f'Context:\n{context}\n'
		f'---\n\n'
		f'Question: {request.query}'
	)
	return RagChatResponse(prompt=augmented_prompt)
