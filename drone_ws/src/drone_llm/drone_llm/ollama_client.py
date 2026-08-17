"""Minimal Ollama chat client.

Deliberately thin: Ollama's /api/chat already speaks tool-calling, so wrapping
it in a heavier framework (langchain/langgraph, as the sibling drone_llm_agent
project does) would add dependencies without buying anything here.
"""

import json

import requests


class OllamaError(RuntimeError):
    """Raised when the model server is unreachable or returns an error.

    The bridge catches this and reports it to the user instead of dying: a
    flying drone must not be taken down by a chat backend hiccup.
    """


class OllamaClient:

    def __init__(self, host='http://localhost:11434', model='qwen3:14b',
                 timeout=120.0):
        self.host = host.rstrip('/')
        self.model = model
        self.timeout = timeout

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=512,
             timeout=None):
        """Send a chat turn. Returns the raw ``message`` dict from Ollama,
        which carries ``content`` and optionally ``tool_calls``."""
        payload = {
            'model': self.model,
            'messages': messages,
            'stream': False,
            # Qwen3 is a hybrid-reasoning model and emits <think>...</think>
            # blocks by default. That would blow the 5s telemetry budget and
            # pollute both the narration text and tool-call parsing, so the
            # reasoning pass is switched off explicitly.
            'think': False,
            'options': {
                'temperature': temperature,
                'num_predict': max_tokens,
            },
        }
        if tools:
            payload['tools'] = tools

        try:
            resp = requests.post(f'{self.host}/api/chat', json=payload,
                                 timeout=timeout or self.timeout)
            resp.raise_for_status()
            return resp.json().get('message', {})
        except requests.exceptions.RequestException as exc:
            raise OllamaError(f'Ollama request failed: {exc}') from exc
        except json.JSONDecodeError as exc:
            raise OllamaError(f'Ollama returned malformed JSON: {exc}') from exc

    def available(self):
        """True if the server is up and the configured model is pulled."""
        try:
            resp = requests.get(f'{self.host}/api/tags', timeout=5.0)
            resp.raise_for_status()
            names = [m.get('name', '') for m in resp.json().get('models', [])]
        except (requests.exceptions.RequestException, json.JSONDecodeError):
            return False
        # Ollama reports "qwen3:14b"; accept a bare "qwen3" config too.
        return any(n == self.model or n.startswith(f'{self.model}:')
                   for n in names)
