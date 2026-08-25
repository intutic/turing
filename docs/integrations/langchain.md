# LangChain & LangGraph Integration Guide

[LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph) support Turing Engine as an inference runtime for agentic workflows, multi-step chains, and tool calling.

---

## 1. Quick Start: Standard OpenAI Client

Turing Engine provides native drop-in compatibility with `langchain-openai`:

```bash
pip install langchain langchain-openai
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Connect to local Turing Engine instance
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="turing-local",
    model="llama-3.1-70b",
    temperature=0.7,
    max_tokens=256
)

messages = [
    SystemMessage(content="You are a helpful software architecture assistant."),
    HumanMessage(content="How does Turing Engine achieve 75% KV cache compression?")
]

response = llm.invoke(messages)
print(response.content)
```

---

## 2. Streaming Responses in LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="turing-local",
    model="qwen-2.5-72b",
    streaming=True
)

for chunk in llm.stream("Write a fast C++20 matrix multiplication function using AVX2 SIMD:"):
    print(chunk.content, end="", flush=True)
```

---

## 3. LangGraph Multi-Agent Workflows

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

class AgentState(TypedDict):
    task: str
    code: str
    review: str

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="turing-local",
    model="llama-3.1-70b"
)

def developer_node(state: AgentState):
    prompt = f"Write Python code to solve: {state['task']}"
    resp = llm.invoke(prompt)
    return {"code": resp.content}

def reviewer_node(state: AgentState):
    prompt = f"Review this Python code for correctness and performance:\n{state['code']}"
    resp = llm.invoke(prompt)
    return {"review": resp.content}

graph = StateGraph(AgentState)
graph.add_node("developer", developer_node)
graph.add_node("reviewer", reviewer_node)
graph.set_entry_point("developer")
graph.add_edge("developer", "reviewer")
graph.add_edge("reviewer", END)

app = graph.compile()
result = app.invoke({"task": "Compute shortest path using Dijkstra with a priority queue."})
print("--- Final Review ---\n", result["review"])
```
