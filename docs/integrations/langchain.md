# 🦜️🔗 LangChain & LangGraph Integration Guide

[LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph) natively integrate with **Turing Engine** for high-throughput agentic workflows, multi-step deliberation, and tool calling.

---

## ⚡ Method 1: Drop-In `langchain-openai` (Zero Extra Setup)

Because Turing Engine provides a standard `/v1/chat/completions` API endpoint, you can use the standard `langchain-openai` package directly:

```bash
pip install langchain-openai
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Connect to local Turing Engine instance
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="turing-local",
    model="deepseek-r1-1.5b",
    temperature=0.7,
    max_tokens=512
)

messages = [
    SystemMessage(content="You are an expert systems engineer."),
    HumanMessage(content="Explain how SVD INT8 KV cache paging achieves 75% memory compression.")
]

response = llm.invoke(messages)
print(response.content)
```

---

## 🚀 Method 2: Native `ChatTuring` (Subspace Controls)

For direct control over **57% Subspace Activation Pruning** and **Rank-64 SVD KV cache** parameters:

```python
from turing.integrations.langchain import ChatTuring

llm = ChatTuring(
    model="deepseek-r1-1.5b",
    base_url="http://localhost:8000/v1",
    sparsity_ratio=0.57, # 57.1% FFN channel pruning
    svd_rank=64          # Rank-64 SVD INT8 KV cache
)

# 1. Synchronous Invocation
response = llm.invoke("Write a C++20 AVX2 SIMD matrix multiplication routine:")
print(response["content"])

# 2. Token Streaming
for chunk in llm.stream("Explain the mathematical formulation of cross-model W* KV transfer:"):
    print(chunk, end="", flush=True)
```

---

## 🧠 Method 3: LangGraph Multi-Agent Workflows

You can coordinate complex multi-agent deliberation workflows in **LangGraph** running on a single consumer GPU:

```python
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from turing.integrations.langchain import ChatTuring

# Define Multi-Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_step: str

llm = ChatTuring(model="deepseek-r1-1.5b", base_url="http://localhost:8000/v1")

# Define Deliberation Agents
def researcher_node(state: AgentState):
    prompt = f"Research technical trade-offs for: {state['messages'][-1].content}"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=f"Researcher: {res['content']}")]}

def reviewer_node(state: AgentState):
    prompt = f"Critique and verify the following analysis: {state['messages'][-1].content}"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=f"Reviewer: {res['content']}")]}

# Build LangGraph State Graph
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("reviewer", reviewer_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "reviewer")
workflow.add_edge("reviewer", END)

app = workflow.compile()

# Execute Multi-Agent Graph
result = app.invoke({"messages": [HumanMessage(content="How to optimize Triton kernels for AMD ROCm?")]})
for msg in result["messages"]:
    print(msg.content)
```
