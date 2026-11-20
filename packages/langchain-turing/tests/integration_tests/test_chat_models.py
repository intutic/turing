"""LangChain's standard integration tests against a live `turing serve` ($TURING_BASE_URL, $TURING_MODEL)."""

import os
from typing import Type

import pytest
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_turing import ChatTuring

pytestmark = pytest.mark.skipif(not os.getenv("TURING_BASE_URL"), reason="set TURING_BASE_URL to a running turing serve")


class TestChatTuringIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatTuring]:
        return ChatTuring

    @property
    def chat_model_params(self) -> dict:
        return {"model": os.getenv("TURING_MODEL", "smollm2"), "temperature": 0, "max_tokens": 256}
