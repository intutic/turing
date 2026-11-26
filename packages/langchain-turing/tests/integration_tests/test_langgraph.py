"""A two-agent LangGraph graph on ChatTuring against a live `turing serve` ($TURING_BASE_URL, $TURING_MODEL): a
researcher that calls a tool through ToolNode, then a writer that answers from the researcher's result."""

import os

import pytest

pytest.importorskip("langgraph")
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_turing import ChatTuring

pytestmark = pytest.mark.skipif(not os.getenv("TURING_BASE_URL"), reason="set TURING_BASE_URL to a running turing serve")


def test_two_agent_graph_with_tool():
    executed = []

    @tool
    def get_order_status(order_id: str) -> str:
        """Look up the shipping status of an order by its id."""
        executed.append(order_id)
        return f"Order {order_id} shipped on 2026-11-20 with carrier DHL."

    llm = ChatTuring(model=os.getenv("TURING_MODEL", "smollm2"), temperature=0, max_tokens=128)
    researcher_llm = llm.bind_tools([get_order_status])

    def researcher(state: MessagesState):
        return {"messages": [researcher_llm.invoke(state["messages"])]}

    def writer(state: MessagesState):
        facts = state["messages"][-1].content
        return {"messages": [llm.invoke([
            SystemMessage("You are a customer support writer. Write one short, polite sentence for the customer."),
            HumanMessage(f"Facts from the researcher: {facts}\nWrite the reply.")])]}

    g = StateGraph(MessagesState)
    g.add_node("researcher", researcher)
    g.add_node("tools", ToolNode([get_order_status]))
    g.add_node("writer", writer)
    g.add_edge(START, "researcher")
    g.add_conditional_edges("researcher", tools_condition, {"tools": "tools", END: "writer"})
    g.add_edge("tools", "researcher")
    g.add_edge("writer", END)

    prompt = "What is the shipping status of order A1234? Use the get_order_status tool."
    steps = list(g.compile().stream({"messages": [HumanMessage(prompt)]}, stream_mode="updates"))
    nodes = [node for step in steps for node in step]
    call = steps[0]["researcher"]["messages"][-1].tool_calls[0]
    answer = steps[-1]["writer"]["messages"][-1].content

    assert (call["name"], call["args"]) == ("get_order_status", {"order_id": "A1234"})
    assert executed == ["A1234"]
    assert nodes == ["researcher", "tools", "researcher", "writer"]
    assert "DHL" in answer  # only the tool's result names the carrier
