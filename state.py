"""
Scholar Lens - State Definition
Shared across ALL nodes in the LangGraph graph.
"""

from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages


class SectionContent(TypedDict):
    """Represents one section of the research paper."""
    title: str           # e.g. "Literature Review"
    content: str         # Written content
    score: float         # Evaluator score 1-10
    attempts: int        # How many times writer retried
    citations: List[str] # Raw citation strings for this section
    feedback: str        # Evaluator feedback — passed to writer on retry


# ─────────────────────────────────────────────────────────
# REDUCER — merges parallel search worker results
# Each worker returns { "Section Title": [...content] }
# Reducer merges all into one dict without overwriting
# ─────────────────────────────────────────────────────────

def keep_last(existing: str, new: str) -> str:
    """
    Reducer for status field.
    Parallel search workers all write status simultaneously.
    Without this reducer LangGraph throws InvalidUpdateError.
    keep_last simply takes the newest value.
    """
    return new


def merge_search_results(existing: dict, new: dict) -> dict:
    """
    Called automatically by LangGraph when parallel
    search_worker_nodes write to search_results.

    Worker 1 returns → { "Introduction": [...] }
    Worker 2 returns → { "Literature Review": [...] }
    Worker 3 returns → { "Methodology": [...] }

    Reducer merges all into:
    {
        "Introduction":      [...],
        "Literature Review": [...],
        "Methodology":       [...]
    }

    Without this reducer, workers would overwrite each other.
    """
    return {**existing, **new}


class ResearchState(TypedDict):

    # ── User Inputs ──────────────────────────────
    topic: str
    citation_style: str             # "APA" | "IEEE" | "MLA"

    # ── Planner Output ───────────────────────────
    outline: List[str]              # ["Introduction", "Literature Review", ...]
    research_angle: str             # Unique thesis/angle planner decides
    key_themes: List[str]           # Consistent themes for writer
    suggested_sources: List[str]    # Search queries from planner

    # ── Search & Scrape Results ──────────────────
    # Annotated with reducer — parallel workers merge safely
    search_results: Annotated[dict, merge_search_results]

    # ── Section Writing ──────────────────────────
    sections: List[SectionContent]  # Grows as each section is written + approved
    current_section_index: int      # Which section is being worked on right now

    # ── Final Assembly ───────────────────────────
    abstract: str                   # Written LAST after all sections done
    final_paper: str                # Full assembled paper
    overall_score: float            # Final editor evaluator score
    editor_attempts: int            # How many times final editor retried

    # ── Citations ────────────────────────────────
    formatted_citations: List[str]  # Formatted in chosen citation_style

    # ── Export ───────────────────────────────────
    pdf_path: Optional[str]
    markdown_path: Optional[str]

    # ── UI / Streaming ───────────────────────────
    messages: Annotated[list, add_messages]  # Streams updates to Gradio UI
    status: Annotated[str, keep_last]        # Parallel-safe — keep last write