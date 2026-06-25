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

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

from src.rag.pipeline import EnterpriseRAGPipeline
from src.utils.config import settings

logger = logging.getLogger(__name__)


# ── State definition ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State passed between nodes in the LangGraph."""
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    route: str
    final_answer: str
    sources: list[str]
    confidence: float


# ── Router agent ──────────────────────────────────────────────────────────────

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query router for an enterprise AI system.
    
Classify the user query into exactly one category:

- "rag": Questions about internal documentation, policies, procedures, architecture
- "sql": Questions about data, metrics, counts, aggregations that need database queries  
- "mcp": Questions about live system state — open JIRA tickets, Confluence pages, emails
- "direct": Greetings, general knowledge, questions clearly outside the enterprise scope

Respond with ONLY the category label, nothing else."""),
    ("human", "Query: {query}")
])


class EnterpriseRouterAgent:
    """
    Multi-agent router using LangGraph.
    
    Every routing decision and subagent call is traced in LangSmith,
    giving full observability into how queries flow through the system.
    """

    def __init__(self):
        self.llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0)
        self.rag_pipeline = EnterpriseRAGPipeline()
        self._router_chain = ROUTER_PROMPT | self.llm | StrOutputParser()
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
        """Node: classify the query and set the route."""
        query = state["query"]
        route = self._router_chain.invoke({"query": query}).strip().lower()

        # Validate route
        valid_routes = {"rag", "sql", "mcp", "direct"}
        if route not in valid_routes:
            logger.warning("Invalid route '%s', defaulting to 'rag'", route)
            route = "rag"

        logger.info("Query routed to: %s", route)
        return {**state, "route": route}

    def _dispatch(self, state: AgentState) -> str:
        """Conditional edge: return the route label."""
        return state.get("route", "direct")

    def _rag_agent(self, state: AgentState) -> AgentState:
        """Node: answer using the RAG pipeline."""
        response = self.rag_pipeline.invoke(state["query"])
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

    def invoke(self, query: str, chat_history: list = None) -> dict:
        """
        Run the full routing pipeline.
        
        Args:
            query: User's question
            chat_history: Previous messages for context
            
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
        }

        final_state = self._graph.invoke(initial_state)

        return {
            "answer": final_state["final_answer"],
            "sources": final_state["sources"],
            "confidence": final_state["confidence"],
            "route": final_state["route"],
        }
