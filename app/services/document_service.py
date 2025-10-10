import chromadb
import io
import os
import pytesseract
import fitz
import ollama
from PIL import Image
from typing import List, Dict, Any, Optional
from fastapi import UploadFile, HTTPException
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from app.schemas.document import Document
from app.core.constants import CHROMA_PATH

client = chromadb.PersistentClient(path=CHROMA_PATH)


def _get_collection(model_name: str):
	return client.get_or_create_collection(name=model_name, metadata={'hnsw:space': 'cosine'})


def add_documents_to_collection(files: List[UploadFile], collection_name: str, chat_id: str) -> None:
	collection = _get_collection(collection_name)
	for file in files:
		temp_file_path = f'/tmp/{file.filename}'
		try:
			with open(temp_file_path, 'wb') as buffer:
				buffer.write(file.file.read())

			if file.filename.lower().endswith('.pdf'):
				# Attempt initial text extraction
				docs = PyMuPDFLoader(temp_file_path).load()

				# Check if any meaningful text was extracted
				all_content = ' '.join([d.page_content for d in docs]).strip()
				if not all_content:
					print(f'No text found in {file.filename} via direct extraction. Attempting OCR...')
					# Attempt OCR
					ocr_text = ''
					pdf_document = fitz.open(temp_file_path)
					for page_num in range(pdf_document.page_count):
						page = pdf_document.load_page(page_num)
						pix = page.get_pixmap()
						img_bytes = pix.pil_tobytes(format='PNG')
						img = Image.open(io.BytesIO(img_bytes))
						ocr_text += pytesseract.image_to_string(img)
					pdf_document.close()

					if ocr_text.strip():
						docs = [LangchainDocument(page_content=ocr_text, metadata={'source': file.filename})]
						print(f'OCR successful for {file.filename}. Extracted {len(ocr_text.strip())} characters.')
					else:
						raise HTTPException(
							status_code=400,
							detail=f'Could not extract any meaningful text from {file.filename} even with OCR.',
						)

				text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
				splits = text_splitter.split_documents(docs)

				if not splits:
					raise HTTPException(
						status_code=400,
						detail=f'No text splits could be generated from {file.filename}. Document might be empty or unparseable.',
					)

				# Manually generate embeddings
				try:
					embeddings = []
					for doc_split in splits:
						response = ollama.embeddings(model=collection_name, prompt=doc_split.page_content)
						embeddings.append(response['embedding'])
				except Exception as e:
					print(f'Failed to create embeddings for {file.filename}: {e}')
					raise HTTPException(status_code=500, detail=f'Failed to create embeddings for {file.filename}.')

				ids = [f'{chat_id}-{file.filename}-{i}' for i, _ in enumerate(splits)]
				collection.add(
					embeddings=embeddings,
					documents=[doc.page_content for doc in splits],
					metadatas=[{'chat_id': chat_id, 'source': file.filename} for _ in splits],
					ids=ids,
				)
				print(
					f"Successfully added {len(splits)} splits from {file.filename} to collection '{collection_name}'."
				)

			else:
				raise HTTPException(
					status_code=400,
					detail=f'Unsupported file type: {file.filename}. Only PDF files are currently supported.',
				)

		except HTTPException as e:
			print(f'Error processing file {file.filename}: {e.detail}')
			raise e
		except Exception as e:
			print(f'Failed to process and add file {file.filename}: {e}')
			raise HTTPException(status_code=500, detail=f'Failed to process file {file.filename}: {e}')
		finally:
			if os.path.exists(temp_file_path):
				os.remove(temp_file_path)


def get_relevant_documents(query: str, collection_name: str, chat_id: Optional[str] = None) -> List[Document]:
	where_clause = {}
	if chat_id:
		where_clause['chat_id'] = chat_id

	try:
		query_embedding = ollama.embeddings(model=collection_name, prompt=query)['embedding']
	except Exception as e:
		print(f'Failed to get embedding from Ollama: {e}')
		return []

	collection = _get_collection(collection_name)
	results = collection.query(
		query_embeddings=[query_embedding], n_results=5, where=where_clause if where_clause else None
	)

	documents: List[Document] = []
	if not results or not results['ids'] or not results['ids'][0]:
		return documents

	result_ids = results['ids'][0]
	# distances = results['distances'][0]
	metadatas = results['metadatas'][0]
	docs = results['documents'][0]

	for i in range(len(result_ids)):
		# if distances[i] < 0.45:
		documents.append(Document(id=result_ids[i], filename=metadatas[i].get('source', 'Unknown'), content=docs[i]))

	return documents


def list_documents(collection_name: Optional[str] = None, chat_id: Optional[str] = None) -> List[Dict[str, Any]]:
	where_clause = {}
	if chat_id:
		where_clause['chat_id'] = chat_id

	collections_to_search = []
	if collection_name:
		collections_to_search.append(_get_collection(collection_name))
	else:
		all_chroma_collections = client.list_collections()
		collections_to_search = [_get_collection(col.name) for col in all_chroma_collections]

	unique_sources = set()
	document_list = []
	for collection in collections_to_search:
		all_docs = collection.get(where=where_clause if chat_id else None)
		if all_docs and all_docs['metadatas']:
			for metadata in all_docs['metadatas']:
				source = metadata.get('source')
				if source and source not in unique_sources:
					unique_sources.add(source)
					document_list.append({'filename': source})
	return document_list


def delete_collection(collection_name: str) -> None:
	try:
		client.delete_collection(name=collection_name)
		print(f"Successfully deleted collection '{collection_name}'.")
	except Exception as e:
		print(f"Failed to delete collection '{collection_name}': {e}")
		raise HTTPException(status_code=500, detail=f'Failed to delete collection: {e}')
