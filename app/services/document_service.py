import io
import tempfile
from typing import Any

import chromadb
import fitz
import pytesseract
from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from app.core.constants import CHROMA_PATH, DEFAULT_LLM_BASE_URL
from app.schemas.document import Document
from app.services.embeddings import embed_texts

client = chromadb.PersistentClient(path=CHROMA_PATH)


def _get_collection(model_name: str):
	return client.get_or_create_collection(name=model_name, metadata={'hnsw:space': 'cosine'})


def _extract_pdf_documents(path: str, filename: str) -> list[LangchainDocument]:
	"""Extracts text from a PDF, falling back to OCR for scanned documents."""
	docs = PyMuPDFLoader(path).load()
	if ' '.join(doc.page_content for doc in docs).strip():
		return docs

	print(f'No text found in {filename} via direct extraction. Attempting OCR...')
	ocr_text = ''
	with fitz.open(path) as pdf_document:
		for page in pdf_document:
			image = Image.open(io.BytesIO(page.get_pixmap().pil_tobytes(format='PNG')))
			ocr_text += pytesseract.image_to_string(image)

	if not ocr_text.strip():
		raise HTTPException(
			status_code=400, detail=f'Could not extract any meaningful text from {filename} even with OCR.'
		)
	print(f'OCR successful for {filename}. Extracted {len(ocr_text.strip())} characters.')
	return [LangchainDocument(page_content=ocr_text, metadata={'source': filename})]


def add_documents_to_collection(
	files: list[UploadFile],
	collection_name: str,
	chat_id: str | None = None,
	provider: str = 'ollama',
	base_url: str = DEFAULT_LLM_BASE_URL,
) -> None:
	collection = _get_collection(collection_name)
	for file in files:
		filename = file.filename or 'document.pdf'
		if not filename.lower().endswith('.pdf'):
			raise HTTPException(
				status_code=400, detail=f'Unsupported file type: {filename}. Only PDF files are currently supported.'
			)
		try:
			with tempfile.NamedTemporaryFile(suffix='.pdf') as temp:
				temp.write(file.file.read())
				temp.flush()
				docs = _extract_pdf_documents(temp.name, filename)

			splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
			if not splits:
				raise HTTPException(
					status_code=400,
					detail=f'No text splits could be generated from {filename}. Document might be empty or unparseable.',
				)

			try:
				embeddings = embed_texts([doc.page_content for doc in splits], collection_name, provider, base_url)
			except Exception as e:
				print(f'Failed to create embeddings for {filename}: {e}')
				raise HTTPException(status_code=500, detail=f'Failed to create embeddings for {filename}.')

			prefix = f'{chat_id}-' if chat_id else ''
			metadatas = [{'source': filename, **({'chat_id': chat_id} if chat_id else {})} for _ in splits]
			ids = [f'{prefix}{filename}-{i}' for i in range(len(splits))]

			collection.add(
				embeddings=embeddings, documents=[doc.page_content for doc in splits], metadatas=metadatas, ids=ids
			)
			print(f"Successfully added {len(splits)} splits from {filename} to collection '{collection_name}'.")
		except HTTPException as e:
			print(f'Error processing file {filename}: {e.detail}')
			raise
		except Exception as e:
			print(f'Failed to process and add file {filename}: {e}')
			raise HTTPException(status_code=500, detail=f'Failed to process file {filename}: {e}')


def get_relevant_documents(
	query: str,
	collection_name: str,
	chat_id: str | None = None,
	provider: str = 'ollama',
	base_url: str = DEFAULT_LLM_BASE_URL,
) -> list[Document]:
	try:
		query_embedding = embed_texts([query], collection_name, provider, base_url)[0]
	except Exception as e:
		print(f'Failed to embed the query with {provider}: {e}')
		return []

	results = _get_collection(collection_name).query(
		query_embeddings=[query_embedding], n_results=5, where={'chat_id': chat_id} if chat_id else None
	)
	if not results or not results['ids'] or not results['ids'][0]:
		return []

	return [
		Document(id=result_id, filename=metadata.get('source', 'Unknown'), content=content)
		for result_id, metadata, content in zip(results['ids'][0], results['metadatas'][0], results['documents'][0])
	]


def list_documents(collection_name: str | None = None, chat_id: str | None = None) -> list[dict[str, Any]]:
	collections = (
		[_get_collection(collection_name)]
		if collection_name
		else [_get_collection(collection.name) for collection in client.list_collections()]
	)

	unique_sources: set[str] = set()
	document_list = []
	for collection in collections:
		all_docs = collection.get(where={'chat_id': chat_id} if chat_id else None)
		for metadata in all_docs['metadatas'] or []:
			source = metadata.get('source')
			if source and source not in unique_sources:
				unique_sources.add(source)
				document_list.append({'filename': source})
	return document_list
