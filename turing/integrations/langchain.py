"""
LangChain and LangGraph integration adapter for Turing Engine.
Provides native ChatTuring and TuringLLM interfaces with Subspace Pruning controls.
"""

from typing import Any, Dict, List, Optional, Iterator, Union
import json
import urllib.request
import urllib.error

class ChatTuring:
    """
    ChatTuring provides native LangChain interface to local or remote Turing Engine.
    Compatible with LangChain Core, LangGraph, and standard tool calling.
    """
    def __init__(
        self,
        model: str = "deepseek-r1-1.5b",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "turing-local",
        temperature: float = 0.7,
        max_tokens: int = 512,
        sparsity_ratio: float = 0.57,
        svd_rank: int = 64,
        timeout: float = 60.0
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.sparsity_ratio = sparsity_ratio
        self.svd_rank = svd_rank
        self.timeout = timeout

    def _format_messages(self, messages: Any) -> List[Dict[str, str]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        
        formatted = []
        for m in messages:
            if hasattr(m, "content"):
                msg_type = getattr(m, "type", "user")
                if msg_type in ("human", "user"):
                    role = "user"
                elif msg_type in ("ai", "assistant"):
                    role = "assistant"
                elif msg_type in ("system",):
                    role = "system"
                elif msg_type in ("tool",):
                    role = "tool"
                else:
                    role = "user"
                formatted.append({"role": role, "content": str(m.content)})
            elif isinstance(m, dict):
                formatted.append(m)
            elif isinstance(m, str):
                formatted.append({"role": "user", "content": m})
        return formatted

    def invoke(self, messages: Any, **kwargs) -> Dict[str, Any]:
        """
        Synchronously invoke the model with input messages.
        """
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False
        }

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "X-Turing-Sparsity": str(self.sparsity_ratio),
                "X-Turing-SVD-Rank": str(self.svd_rank)
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            
            content = data["choices"][0]["message"]["content"]
            return {"content": content, "raw": data}
        except urllib.error.URLError as e:
            # Provide helpful diagnostics if server is offline
            raise ConnectionError(
                f"Failed to connect to Turing Engine at {self.base_url}. "
                f"Ensure 'turing serve --model {self.model}' is running. Error: {e}"
            )

    def stream(self, messages: Any, **kwargs) -> Iterator[str]:
        """
        Streams model completion tokens incrementally.
        """
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True
        }

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "X-Turing-Sparsity": str(self.sparsity_ratio)
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: ") and line_str != "data: [DONE]":
                        chunk = json.loads(line_str[6:])
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Turing Engine at {self.base_url}: {e}")

class TuringLLM:
    """
    Text completion LLM interface for standard LangChain pipelines.
    """
    def __init__(
        self,
        model: str = "deepseek-r1-1.5b",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "turing-local",
        temperature: float = 0.7,
        max_tokens: int = 512,
        sparsity_ratio: float = 0.57
    ):
        self.chat = ChatTuring(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            sparsity_ratio=sparsity_ratio
        )

    def __call__(self, prompt: str, **kwargs) -> str:
        res = self.chat.invoke(prompt, **kwargs)
        return res["content"]

ChatTuringEngine = ChatTuring

__all__ = ["ChatTuring", "ChatTuringEngine", "TuringLLM"]
