"""
Scholar Lens - Graph Builder
Defines all nodes, edges, conditional routing, and checkpointing.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.constants import Send
import os

from state import ResearchState
from agents.planner import planner_node
from agents.search import search_worker_node
from agents.writer import writer_node
from agents.evaluator import evaluator_node, final_evaluator_node
from agents.citation import citation_node
from agents.abstract_writer import abstract_node
from agents.final_editor import final_editor_node
from agents.exporter import exporter_node


# ─────────────────────────────────────────────────────────
# PARALLEL DISPATCH — Send() API
# Spawns one search_worker_node per section simultaneously
# ─────────────────────────────────────────────────────────

def dispatch_searches(state: ResearchState) -> list[Send]:
    """
    Called as conditional edge after planner interrupt is resumed.

    Instead of one search node looping sections sequentially,
    Send() spawns an independent search_worker_node per section.
    All workers run in parallel — LangGraph manages concurrency.

    Each Send() passes a mini-state dict to the worker:
      - section_title    → which section to search for
      - topic            → for query building
      - research_angle   → for richer queries
      - key_themes       → for richer queries
      - suggested_sources → planner's suggested queries

    Results merged back via merge_search_results reducer in state.py
    """
    return [
        Send("search_worker_node", {
            "section_title":     section,
            "topic":             state["topic"],
            "research_angle":    state["research_angle"],
            "key_themes":        state["key_themes"],
            "suggested_sources": state["suggested_sources"],
        })
        for section in state["outline"]
    ]


# ─────────────────────────────────────────────────────────
# CONDITIONAL EDGE: Section Evaluator → Writer or Next
# ─────────────────────────────────────────────────────────

def route_section_evaluator(state: ResearchState) -> str:
    """
    After evaluator scores a section:
    - score < 7 AND attempts < 3 → retry writer
    - score < 7 AND attempts >= 3 → move on anyway (avoid infinite loop)
    - score >= 7 → advance to next section or citations
    """
    sections        = state["sections"]
    current_idx     = state["current_section_index"]
    current_section = sections[current_idx]

    # Cap retries at 3 — move on regardless after 3 attempts
    # Prevents infinite loops when evaluator keeps scoring below threshold
    if current_section["score"] < 7.0 and current_section["attempts"] < 3:
        return "writer_node"

    # Move on — either passed or hit max attempts
    if current_idx + 1 < len(state["outline"]):
        return "advance_section"
    else:
        return "citation_node"


def advance_section(state: ResearchState) -> dict:
    """Increments current_section_index to process next section."""
    return {"current_section_index": state["current_section_index"] + 1}


# ─────────────────────────────────────────────────────────
# CONDITIONAL EDGE: Final Evaluator → Editor or Export
# ─────────────────────────────────────────────────────────

def route_final_evaluator(state: ResearchState) -> str:
    """
    After final paper is scored:
    - score < 8 → back to final_editor_node
    - score >= 8 → export
    """
    if state["overall_score"] < 7.0:
        return "final_editor_node"
    return "exporter_node"


# ─────────────────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────────────────

def build_graph():
    """
    Builds and compiles the full Scholar Lens LangGraph.

    LangGraph features used:
      StateGraph          — main graph definition
      TypedDict state     — shared whiteboard
      Annotated reducer   — merge_search_results for parallel workers
      Send() API          — parallel search worker dispatch
      Conditional edges   — evaluator routing, final evaluator routing
      interrupt_after     — human-in-the-loop after planner
      SqliteSaver         — checkpointing, resume across sessions
      stream_mode=values  — streaming to Gradio UI
    """

    builder = StateGraph(ResearchState)

    # ── Register Nodes ──────────────────────────────────
    builder.add_node("planner_node",         planner_node)
    builder.add_node("search_worker_node",   search_worker_node)  # parallel worker
    builder.add_node("writer_node",          writer_node)
    builder.add_node("evaluator_node",       evaluator_node)
    builder.add_node("advance_section",      advance_section)
    builder.add_node("citation_node",        citation_node)
    builder.add_node("abstract_node",        abstract_node)
    builder.add_node("final_editor_node",    final_editor_node)
    builder.add_node("final_evaluator_node", final_evaluator_node)
    builder.add_node("exporter_node",        exporter_node)

    # ── START → Planner ─────────────────────────────────
    builder.add_edge(START, "planner_node")

    # ── Planner → Parallel Search Dispatch ──────────────
    # Graph pauses after planner (interrupt_after)
    # When user resumes, dispatch_searches fires Send() per section
    # All search_worker_nodes run simultaneously
    builder.add_conditional_edges(
        "planner_node",
        dispatch_searches,
        ["search_worker_node"]   # all Send() targets this node
    )

    # ── Search Workers → Writer ──────────────────────────
    # All parallel workers must finish before writer starts
    # LangGraph waits for all Send() workers automatically
    builder.add_edge("search_worker_node", "writer_node")

    # ── Writer → Evaluator ──────────────────────────────
    builder.add_edge("writer_node", "evaluator_node")

    # ── Evaluator → (retry | advance | citations) ───────
    builder.add_conditional_edges(
        "evaluator_node",
        route_section_evaluator,
        {
            "writer_node":     "writer_node",
            "advance_section": "advance_section",
            "citation_node":   "citation_node",
        }
    )

    # ── Advance Section → Writer ─────────────────────────
    builder.add_edge("advance_section", "writer_node")

    # ── Citations → Abstract → Final Editor ─────────────
    builder.add_edge("citation_node",      "abstract_node")
    builder.add_edge("abstract_node",      "final_editor_node")
    builder.add_edge("final_editor_node",  "final_evaluator_node")

    # ── Final Evaluator → (retry editor | export) ───────
    builder.add_conditional_edges(
        "final_evaluator_node",
        route_final_evaluator,
        {
            "final_editor_node": "final_editor_node",
            "exporter_node":     "exporter_node",
        }
    )

    # ── Export → END ────────────────────────────────────
    builder.add_edge("exporter_node", END)

    # ── SqliteSaver Checkpointer ────────────────────────
    # Persists graph state after every node to SQLite
    # User can reload page → BrowserState restores thread_id
    # → SQLite loads checkpoint → session resumes exactly
    # /tmp is writable on both local dev and HuggingFace Docker
    try:
    
        DB_PATH = os.getenv("DB_PATH", "scholar_lens.db")

        conn = sqlite3.connect(
    DB_PATH,
    check_same_thread=False,
    isolation_level=None   # ✅ better concurrency
)
    
        checkpointer = SqliteSaver(conn)
    except Exception:
        # Fallback to in-memory if SQLite fails
        checkpointer = MemorySaver()

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=["planner_node"],
    )


# Singleton graph instance
graph = build_graph()