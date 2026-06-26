"""
src/agents/router.py
---------------------
LangGraph-based routing agent that directs queries to the right specialist.

This is the A2A (Agent-to-Agent) pattern Syama built in OpsFlow,
now applied to a RAG context. The router decides:
- RAG Agent: knowledge base questions
- SQL Agent: structured data queries  
- MCP Agent: live system data (JIRA, Confluence)
- Direct LLM: general questions, greetings

LangSmith traces the entire routing decision + subagent execution.
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import logging

from src.rag.pipeline import EnterpriseRAGPipeline
from src.utils.config import settings

logger = logging.getLogger(__name__)


# ── Keyword router (no LLM call — saves 1-2s per query) ──────────────────────

_RAG_KEYWORDS = {
    "policy", "policies", "procedure", "procedures", "retention", "compliance",
    "regulation", "approved", "approval", "requirement", "requirements",
    "guideline", "guidelines", "rule", "rules", "standard", "standards",
    "security", "pii", "gdpr", "hipaa", "soc2", "encryption", "access",
    "permission", "authentication", "authorization", "incident", "breach",
    "aws", "region", "cloud", "backup", "storage", "vendor", "contract",
    "onboarding", "offboarding", "data", "classification", "handling",
}

_SQL_KEYWORDS = {"how many", "count", "total", "average", "avg", "sum",
                 "statistics", "metrics", "report", "trend", "percentage"}

_MCP_KEYWORDS = {"jira", "confluence", "ticket", "issue #", "sprint",
                 "kanban", "open ticket", "open issue"}


def keyword_route(query: str) -> str:
    """Route a query using keyword matching — zero LLM calls, ~0ms."""
    q = query.lower()
    if any(kw in q for kw in _MCP_KEYWORDS):
        return "mcp"
    if any(kw in q for kw in _SQL_KEYWORDS):
        return "sql"
    if any(kw in q for kw in _RAG_KEYWORDS):
        return "rag"
    return "direct"


# ── State definition ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State passed between nodes in the LangGraph."""
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    route: str
    final_answer: str
    sources: list[str]
    confidence: float
    rewrite_query: bool
    check_faithfulness: bool


class EnterpriseRouterAgent:
    """
    Multi-agent router using LangGraph.
    
    Every routing decision and subagent call is traced in LangSmith,
    giving full observability into how queries flow through the system.
    """

    def __init__(self):
        self.llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0)
        self.rag_pipeline = EnterpriseRAGPipeline()
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph routing graph."""
        graph = StateGraph(AgentState)

        # Nodes
        graph.add_node("router", self._route_query)
        graph.add_node("rag_agent", self._rag_agent)
        graph.add_node("mcp_agent", self._mcp_agent)
        graph.add_node("direct_agent", self._direct_agent)

        # Edges: router decides next node
        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            self._dispatch,
            {
                "rag": "rag_agent",
                "mcp": "mcp_agent",
                "direct": "direct_agent",
                "sql": "direct_agent",  # SQL agent placeholder
            }
        )

        # All agents end after their response
        graph.add_edge("rag_agent", END)
        graph.add_edge("mcp_agent", END)
        graph.add_edge("direct_agent", END)

        return graph.compile()

    def _route_query(self, state: AgentState) -> AgentState:
        """Node: classify the query using keyword matching (zero LLM calls)."""
        route = keyword_route(state["query"])
        logger.info("Query keyword-routed to: %s", route)
        return {**state, "route": route}

    def _dispatch(self, state: AgentState) -> str:
        """Conditional edge: return the route label."""
        return state.get("route", "direct")

    def _rag_agent(self, state: AgentState) -> AgentState:
        """Node: answer using the RAG pipeline."""
        response = self.rag_pipeline.invoke(
            state["query"],
            rewrite_query=state.get("rewrite_query", True),
            check_faithfulness=state.get("check_faithfulness", True),
        )
        return {
            **state,
            "final_answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence,
            "messages": state["messages"] + [AIMessage(content=response.answer)]
        }

    def _mcp_agent(self, state: AgentState) -> AgentState:
        """
        Node: answer using live system data via MCP.
        In production this calls the MCP server (JIRA, Confluence).
        """
        # Placeholder — in production, calls MCP server
        answer = (
            f"[MCP Agent] Query routed to live system lookup. "
            f"In production, this fetches real-time data from JIRA/Confluence "
            f"for: '{state['query']}'"
        )
        return {
            **state,
            "final_answer": answer,
            "sources": ["JIRA", "Confluence"],
            "confidence": 0.9,
            "messages": state["messages"] + [AIMessage(content=answer)]
        }

    def _direct_agent(self, state: AgentState) -> AgentState:
        """Node: answer directly with the LLM (no retrieval needed)."""
        response = self.llm.invoke(state["messages"] + [HumanMessage(content=state["query"])])
        answer = response.content
        return {
            **state,
            "final_answer": answer,
            "sources": [],
            "confidence": 0.8,
            "messages": state["messages"] + [AIMessage(content=answer)]
        }

    def invoke(
        self,
        query: str,
        chat_history: list = None,
        rewrite_query: bool = True,
        check_faithfulness: bool = True,
    ) -> dict:
        """
        Run the full routing pipeline.

        Args:
            query: User's question
            chat_history: Previous messages for context
            rewrite_query: Rewrite query before retrieval (saves ~1-2s if False)
            check_faithfulness: Run LLM faithfulness guardrail (saves ~1-2s if False)

        Returns:
            dict with answer, sources, confidence, and route taken
        """
        initial_state: AgentState = {
            "messages": chat_history or [],
            "query": query,
            "route": "",
            "final_answer": "",
            "sources": [],
            "confidence": 0.0,
            "rewrite_query": rewrite_query,
            "check_faithfulness": check_faithfulness,
        }

        final_state = self._graph.invoke(initial_state)

        return {
            "answer": final_state["final_answer"],
            "sources": final_state["sources"],
            "confidence": final_state["confidence"],
            "route": final_state["route"],
        }

    async def astream(
        self,
        query: str,
        top_k: int = 5,
        rewrite_query: bool = False,
        check_faithfulness: bool = False,
    ):
        """
        Stream tokens for a query. Yields SSE-ready dicts:
          {"type": "token", "content": "..."}   — one per LLM token
          {"type": "done",  "sources": [...], "confidence": ..., "route": "..."}
        """
        route = keyword_route(query)
        logger.info("Stream query keyword-routed to: %s", route)

        if route == "rag":
            async for event in self.rag_pipeline.astream(
                query, top_k=top_k, rewrite_query=rewrite_query,
                check_faithfulness=check_faithfulness,
            ):
                yield event
        else:
            # Direct LLM — stream without retrieval
            async for chunk in self.llm.astream([HumanMessage(content=query)]):
                yield {"type": "token", "content": chunk.content}
            yield {"type": "done", "sources": [], "confidence": 0.8, "route": route}
