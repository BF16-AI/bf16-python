"""Official zero-dependency Python SDK for bf16.ai (https://bf16.ai).

Uncensored 1M-context open-weight LLM inference API (DeepSeek V4.1 Flash,
GLM-5.3, GLM-5.3 Flash, Kimi K3.1 Pro) with automatic prompt caching and
strict Zero Data Retention (ZDR).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

__version__ = "0.1.0"

DEFAULT_BASE_URL = "https://api.bf16.ai/v1"
ANTHROPIC_BASE_URL = "https://api.bf16.ai"

MODEL_GLM_53 = "zai-org/glm-5.3-uncensored"
MODEL_GLM_53_FLASH = "zai-org/glm-5.3-flash-uncensored"
MODEL_DEEPSEEK_V41_FLASH = "deepseek/deepseek-v4.1-flash-uncensored"
MODEL_DEEPSEEK_V41_FLASH_FAST = "deepseek/deepseek-v4.1-flash-uncensored-fast"
MODEL_KIMI_K31_PRO = "moonshotai/kimi-k3.1-pro-uncensored"
MODEL_AUTO = "bf16/auto"


class Bf16Error(Exception):
    """Error raised when the bf16.ai API returns an HTTP or transport error."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class Client:
    """Lightweight zero-dependency client for https://api.bf16.ai/v1."""

    api_key: Optional[str] = None
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 120.0
    user_agent: str = f"bf16-python/{__version__}"

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("BF16_API_KEY", "").strip()
        else:
            self.api_key = self.api_key.strip()
        self.base_url = self.base_url.rstrip("/")

    def _headers(self, require_auth: bool = True) -> Dict[str, str]:
        headers = {
            "User-Agent": self.user_agent,
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif require_auth:
            raise Bf16Error("BF16_API_KEY is required for this endpoint.")
        return headers

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        require_auth: bool = True,
    ) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=self._headers(require_auth=require_auth),
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise Bf16Error(f"bf16 API error ({exc.code}): {body}", status_code=exc.code) from exc
        except urllib.error.URLError as exc:
            raise Bf16Error(f"bf16 transport error: {exc.reason}") from exc

    def list_models(self) -> List[Dict[str, Any]]:
        """Fetch active 1M-context models and live pricing from GET /v1/models (no key needed)."""
        res = self._request_json("GET", "/models", require_auth=False)
        return list(res.get("data", []))

    def me(self) -> Dict[str, Any]:
        """Fetch live USD balance, tier, and concurrency limits from GET /v1/me."""
        return self._request_json("GET", "/me", require_auth=True)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: str = MODEL_DEEPSEEK_V41_FLASH_FAST,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Execute POST /v1/chat/completions and return the parsed JSON response."""
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            **extra,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        return self._request_json("POST", "/chat/completions", payload=body, require_auth=True)

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: str = MODEL_DEEPSEEK_V41_FLASH_FAST,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **extra: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Stream parsed SSE JSON chunks from POST /v1/chat/completions."""
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            **extra,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature

        headers = self._headers(require_auth=True)
        headers["Accept"] = "text/event-stream"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise Bf16Error(f"bf16 API error ({exc.code}): {err_body}", status_code=exc.code) from exc

        with resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk_str = line[5:].strip()
                if not chunk_str or chunk_str == "[DONE]":
                    continue
                try:
                    yield json.loads(chunk_str)
                except json.JSONDecodeError:
                    continue


__all__ = [
    "ANTHROPIC_BASE_URL",
    "Bf16Error",
    "Client",
    "DEFAULT_BASE_URL",
    "MODEL_AUTO",
    "MODEL_DEEPSEEK_V41_FLASH",
    "MODEL_DEEPSEEK_V41_FLASH_FAST",
    "MODEL_GLM_53",
    "MODEL_GLM_53_FLASH",
    "MODEL_KIMI_K31_PRO",
    "__version__",
]
