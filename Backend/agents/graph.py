r"""
Assemble the LangGraph agent pipeline.

Graph topology:

    query_analyzer
          |
       router
    /    |      \
simple analytics advanced
    \     |      /
      synthesis
          |
         END
"""
from __future__ import annotations
from langgraph.graph import StateGraph, END

from .state import AgentState
from .query_analyzer import query_analyzer_node
from .router import router_node, route_decision
from .retrieval import (
    simple_retrieval_node,
    advanced_retrieval_node,
    analytics_retrieval_node,
)
from .synthesis import synthesis_node


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("query_analyzer", query_analyzer_node)
    graph.add_node("router", router_node)
    graph.add_node("simple_retrieval", simple_retrieval_node)
    graph.add_node("analytics_retrieval", analytics_retrieval_node)
    graph.add_node("advanced_retrieval", advanced_retrieval_node)
    graph.add_node("synthesis", synthesis_node)

    # Entry
    graph.set_entry_point("query_analyzer")

    # Linear edges
    graph.add_edge("query_analyzer", "router")

    # Conditional fork
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "simple": "simple_retrieval",
            "analytics": "analytics_retrieval",
            "advanced": "advanced_retrieval",
        },
    )

    # Merge back to synthesis
    graph.add_edge("simple_retrieval", "synthesis")
    graph.add_edge("analytics_retrieval", "synthesis")
    graph.add_edge("advanced_retrieval", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


# Module-level compiled graph – import this in other modules
rag_graph = build_graph()
