"""
LlamaIndex on Turing Engine: llama-index's own OpenAILike client pointed at `turing serve`.

There is no separate llama-index package for Turing; any OpenAI-compatible client works.

    pip install llama-index-llms-openai-like
    turing serve --model smollm2 --port 8000
    python integrations/llamaindex/openai_like_example.py
"""

from llama_index.llms.openai_like import OpenAILike


def turing_llm(model: str = "smollm2", api_base: str = "http://localhost:8000/v1", **kwargs) -> OpenAILike:
    return OpenAILike(model=model, api_base=api_base, api_key="turing-local", is_chat_model=True, **kwargs)


if __name__ == "__main__":
    print(turing_llm(max_tokens=64).complete("Explain KV cache paging in one sentence:").text)
