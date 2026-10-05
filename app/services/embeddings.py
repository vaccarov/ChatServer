"""Provider-agnostic text embeddings.

Ollama exposes its native `POST /api/embed`, LM Studio and every other
OpenAI-compatible server expose `POST /v1/embeddings`.
"""

import httpx

_TIMEOUT = 300.0


def _v1_root(base_url: str) -> str:
	"""Strips a trailing `/api`, `/v1` or `/api/v1` so every spelling of the server URL works."""
	root = base_url.rstrip('/')
	for suffix in ('/api/v1', '/api', '/v1'):
		if root.endswith(suffix):
			return root[: -len(suffix)]
	return root


def embed_texts(texts: list[str], model: str, provider: str, base_url: str) -> list[list[float]]:
	"""Returns one embedding vector per input text."""
	if provider == 'ollama':
		response = httpx.post(
			f'{_v1_root(base_url)}/api/embed', json={'model': model, 'input': texts}, timeout=_TIMEOUT
		)
		response.raise_for_status()
		return response.json()['embeddings']

	response = httpx.post(
		f'{_v1_root(base_url)}/v1/embeddings', json={'model': model, 'input': texts}, timeout=_TIMEOUT
	)
	response.raise_for_status()
	return [item['embedding'] for item in response.json()['data']]


if __name__ == '__main__':
	# The suffix order matters: '/api/v1' must win over '/api'.
	assert _v1_root('http://h:1234') == 'http://h:1234'
	assert _v1_root('http://h:1234/v1/') == 'http://h:1234'
	assert _v1_root('http://h:11434/api') == 'http://h:11434'
	assert _v1_root('http://h:11434/api/v1') == 'http://h:11434'
	assert _v1_root('http://h/ollama') == 'http://h/ollama'
	print('embeddings self-check OK')
