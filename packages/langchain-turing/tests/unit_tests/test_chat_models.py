"""LangChain's standard unit tests for chat models (no server needed)."""

from typing import Type

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_turing import ChatTuring


class TestChatTuringUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatTuring]:
        return ChatTuring

    @property
    def chat_model_params(self) -> dict:
        return {"model": "Qwen/Qwen2.5-1.5B-Instruct", "base_url": "http://localhost:8000/v1"}

    @property
    def init_from_env_params(self):
        return ({"TURING_BASE_URL": "http://gpu-host:8000/v1", "TURING_API_KEY": "k"}, {"model": "m"},
                {"base_url": "http://gpu-host:8000/v1", "api_key": "k"})
