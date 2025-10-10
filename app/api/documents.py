from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services import document_service
from app.schemas.document import Document, RagChatRequest, RagChatResponse, SearchRequest

router = APIRouter()


@router.post('/upload', summary='Upload documents for RAG')
async def upload_documents(
	files: List[UploadFile] = File(...), embedding_model: str = Form(...), chat_id: str = Form(...)
) -> Dict[str, Any]:
	"""
	Uploads one or more documents, processes them, and adds them to a
	vector collection specific to the chosen embedding model.
	"""
	try:
		document_service.add_documents_to_collection(files=files, collection_name=embedding_model, chat_id=chat_id)
		return {
			'message': f"Successfully uploaded {len(files)} file(s) to collection '{embedding_model}' for chat '{chat_id}'."
		}
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=f'Failed to upload documents: {e}')


@router.get('/list', summary='List documents in a RAG collection for a chat')
async def list_documents_in_collection(
	embedding_model: Optional[str] = None, chat_id: Optional[str] = None
) -> List[Dict[str, Any]]:
	"""
	Lists all unique documents. Can be filtered by collection (embedding_model)
	and/or chat_id.
	"""
	documents = document_service.list_documents(collection_name=embedding_model, chat_id=chat_id)
	return documents


@router.post('/search', summary='Search for documents')
def search_documents(request: SearchRequest) -> List[Document]:
	relevant_docs = document_service.get_relevant_documents(
		query=request.query, collection_name=request.embedding_model, chat_id=request.chat_id
	)
	return relevant_docs


@router.post('/rag_chat', response_model=RagChatResponse, summary='Get augmented prompt for RAG chat')
def rag_chat(request: RagChatRequest) -> RagChatResponse:
	"""
	Performs a RAG search to find relevant documents and returns an augmented prompt.
	"""
	relevant_docs = document_service.get_relevant_documents(
		query=request.query, collection_name=request.embedding_model, chat_id=request.chat_id
	)
	if not relevant_docs:
		return RagChatResponse(prompt=request.query)
	context = '\n\n'.join([doc.content for doc in relevant_docs])
	augmented_prompt = (
		f'Using the following context, please answer the question.\n\n'
		f'---\n'
		f'Context:\n{context}\n'
		f'---\n\n'
		f'Question: {request.query}'
	)
	return RagChatResponse(prompt=augmented_prompt)


@router.delete('/reset/{embedding_model}', summary='Delete a specific document collection')
def delete_collection_endpoint(embedding_model: str) -> Dict[str, str]:
	"""
	Deletes a ChromaDB collection associated with a specific embedding model.
	"""
	try:
		document_service.delete_collection(collection_name=embedding_model)
		return {'message': f"Collection '{embedding_model}' deleted successfully."}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f'Failed to delete collection: {e}')
