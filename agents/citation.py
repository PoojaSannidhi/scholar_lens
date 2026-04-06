"""
Scholar Lens - Citation Agent

Runs after all sections are written and approved.
Collects raw citation strings from every section,
deduplicates them, and formats into chosen citation style.

Reads from state  → sections (citations field), citation_style
Writes to state   → formatted_citations: List[str]

Uses GPT-4o-mini — precise formatting task, not creative writing.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import ResearchState
import os


# ─────────────────────────────────────────────────────────
# STRUCTURED OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────

class FormattedCitations(BaseModel):
    """Structured list of formatted citations."""

    citations: list[str] = Field(
        description="""List of fully formatted citation strings.
        Each citation must be complete and correctly formatted
        for the requested citation style (APA, IEEE, or MLA).

        APA example:
        Brown, T., & Smith, J. (2023). Large language models in software
        engineering. Journal of AI Research, 45(2), 112-134.
        https://doi.org/10.xxxx

        IEEE example:
        T. Brown and J. Smith, "Large language models in software
        engineering," J. AI Res., vol. 45, no. 2, pp. 112-134, 2023.

        MLA example:
        Brown, Tom, and John Smith. "Large Language Models in Software
        Engineering." Journal of AI Research, vol. 45, no. 2, 2023,
        pp. 112-134."""
    )

    total_count: int = Field(
        description="Total number of unique citations formatted."
    )

    style_used: str = Field(
        description="The citation style used: APA, IEEE, or MLA."
    )


# ─────────────────────────────────────────────────────────
# CITATION NODE
# ─────────────────────────────────────────────────────────

def citation_node(state: ResearchState) -> dict:
    """
    Collects all raw citations from every written section,
    deduplicates, and formats them in the chosen citation style.

    Raw citations look like:
      "Author/Title | URL | Year if available"
      (written by Writer Agent)

    Formatted citations look like proper APA/IEEE/MLA strings.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,          # Formatting must be deterministic
        api_key=os.getenv("OPENAI_API_KEY")
    )

    structured_llm = llm.with_structured_output(FormattedCitations)

    # ── Collect all raw citations from every section ─────
    all_raw_citations = []
    for section in state["sections"]:
        all_raw_citations.extend(section.get("citations", []))

    # ── Deduplicate — same URL may appear in multiple sections ──
    # Simple dedup by converting to set then back to list
    unique_citations = list(dict.fromkeys(all_raw_citations))

    if not unique_citations:
        # No citations found — return empty gracefully
        return {
            "formatted_citations": [],
            "status": "⚠️ No citations found in written sections.",
            "messages": [{"role": "assistant", "content": "⚠️ No citations found."}]
        }

    citation_style = state["citation_style"]   # "APA" | "IEEE" | "MLA"

    messages = [
        SystemMessage(content=f"""You are an expert academic citation formatter.
        Your job is to take raw citation data and format it perfectly in {citation_style} style.

        Rules:
        - Format every citation completely — no missing fields
        - If a field is missing (e.g. no author), use best academic practice 
          (e.g. use website name or "Anonymous")
        - If year is missing, use "n.d." (no date) for APA
        - Order citations alphabetically by first author's last name
        - Deduplicate — if same source appears twice, keep only one
        - Every citation must be on its own line
        - Do not number the citations — just format them cleanly"""),

        HumanMessage(content=f"""
        Citation Style: {citation_style}
        Total raw citations to format: {len(unique_citations)}

        === RAW CITATIONS ===
        {chr(10).join(f"{i+1}. {cite}" for i, cite in enumerate(unique_citations))}

        Please format all of these into proper {citation_style} citations.
        Deduplicate if any appear to reference the same source.
        """)
    ]

    result: FormattedCitations = structured_llm.invoke(messages)

    status_msg = (
        f"📚 Citations formatted — "
        f"{result.total_count} unique references in {result.style_used} style."
    )

    return {
        "formatted_citations": result.citations,
        "status": status_msg,
        "messages": [{"role": "assistant", "content": status_msg}]
    }