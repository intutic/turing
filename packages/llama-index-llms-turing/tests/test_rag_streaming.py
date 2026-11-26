"""Vector-store RAG with streaming generation against a live `turing serve` at $TURING_API_BASE serving $TURING_MODEL:
documents embedded with the server's /v1/embeddings, the answer streamed by Turing."""

import os

import pytest

pytest.importorskip("llama_index.embeddings.openai_like")
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from llama_index.llms.turing import Turing

DOCS = [
    "The Eiffel Tower in Paris was completed in 1889 for the World's Fair.",
    "Photosynthesis lets plants turn sunlight, water and carbon dioxide into glucose and oxygen.",
    "The Pacific is the largest and deepest of Earth's oceans.",
    "Zorblax Station, the research base on Mars, keeps its spare oxygen tanks in Hangar 7.",
    "Mount Everest is the highest mountain above sea level, at 8,849 metres.",
    "Python was created by Guido van Rossum and first released in 1991.",
    "The violin has four strings tuned in perfect fifths: G, D, A and E.",
    "Honey never spoils; edible honey has been found in ancient Egyptian tombs.",
]


@pytest.mark.skipif(not os.getenv("TURING_API_BASE"), reason="set TURING_API_BASE to a running turing serve")
def test_vector_store_rag_streaming():
    model, base = os.getenv("TURING_MODEL", "smollm2"), os.environ["TURING_API_BASE"]
    embed = OpenAILikeEmbedding(model_name=model, api_base=base, api_key=os.getenv("TURING_API_KEY", "turing-local"))
    index = VectorStoreIndex.from_documents([Document(text=t) for t in DOCS], embed_model=embed)
    engine = index.as_query_engine(streaming=True, llm=Turing(model=model, temperature=0, max_tokens=64))

    resp = engine.query("Where does Zorblax Station keep its spare oxygen tanks?")
    chunks = [c for c in resp.response_gen if c]

    assert resp.source_nodes[0].node.get_content() == DOCS[3]
    assert len(chunks) >= 2
    assert "Hangar 7" in "".join(chunks)
