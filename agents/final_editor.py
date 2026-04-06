"""
Scholar Lens - Final Editor Agent

Runs after abstract is written.
Assembles all parts into one complete paper and polishes it:
  - Checks flow and consistency between sections
  - Removes repetition across sections
  - Ensures research_angle runs consistently throughout
  - Ensures key_themes appear consistently
  - Fixes academic tone issues

Reads from state  → sections, abstract, formatted_citations,
                    research_angle, key_themes, topic, citation_style
Writes to state   → final_paper: str

Uses GPT-4o-mini — precision editing task.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import ResearchState
import os


# ─────────────────────────────────────────────────────────
# ASSEMBLE PAPER HELPER
# Combines all parts into one raw string before editing
# ─────────────────────────────────────────────────────────

def _assemble_raw_paper(state: ResearchState) -> str:
    """
    Combines abstract + all sections + citations into one raw string.
    Final Editor receives this and returns a polished version.
    """

    # Abstract (already includes keywords line)
    abstract_block = f"## Abstract\n\n{state['abstract']}"

    # All body sections in order
    sections_block = "\n\n".join([
        f"## {section['title']}\n\n{section['content']}"
        for section in state["sections"]
    ])

    # References
    citations = state.get("formatted_citations", [])
    if citations:
        references_block = "## References\n\n" + "\n\n".join(citations)
    else:
        references_block = "## References\n\nNo references available."

    # Assemble in correct academic paper order
    return f"""# {state['topic']}

{abstract_block}

---

{sections_block}

---

{references_block}"""


# ─────────────────────────────────────────────────────────
# FINAL EDITOR NODE
# ─────────────────────────────────────────────────────────

def final_editor_node(state: ResearchState) -> dict:
    """
    Assembles and polishes the complete paper.
    If this is a retry (final_evaluator scored < 8),
    it receives the already assembled paper and improves it further.
    """

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,        # Mostly deterministic — editing not rewriting
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=4096          # Full paper needs high token limit
    )



    editor_attempts = state.get("editor_attempts", 0)
    key_themes      = state.get("key_themes", [])
    themes_str      = ", ".join(key_themes) if key_themes else "Not specified"

    # First attempt — assemble from parts
    # Retry — improve already assembled paper
    if editor_attempts == 0:
        paper_to_edit = _assemble_raw_paper(state)
        task_instruction = "Assemble and polish this paper into its final form."
    else:
        paper_to_edit = state.get("final_paper", _assemble_raw_paper(state))
        task_instruction = (
            f"This paper scored below 8/10 on final evaluation "
            f"(attempt {editor_attempts + 1}). "
            f"Improve it further — focus on coherence, flow, and consistency."
        )

    messages = [
        SystemMessage(content=f"""You are a senior academic editor doing final polish 
        on a research paper before publication. Your job is to:

        1. Assemble all parts in correct academic order
        2. Remove repetition — same idea should not appear twice
        3. Improve transitions between sections
        4. Ensure the research angle runs consistently throughout
        5. Fix any informal language or tone inconsistencies
        6. Ensure key themes appear consistently: {themes_str}

        Do NOT rewrite sections from scratch — polish and connect them.
        Preserve the author's voice and all citations.
        The paper must read as one coherent document, not separate pieces."""),

        HumanMessage(content=f"""
        Topic: {state['topic']}
        Research Angle: {state['research_angle']}
        Key Themes: {themes_str}
        Citation Style: {state['citation_style']}

        Task: {task_instruction}

        === PAPER TO EDIT ===
        {paper_to_edit}
        """)
    ]

    # Use direct invoke — no structured output
    # Structured output JSON overhead eats into token budget for long papers
    response = llm.invoke(messages)
    final_paper = response.content.strip()

    status_msg = f"✨ Final edit complete (attempt {editor_attempts + 1})."

    return {
        "final_paper":     final_paper,
        "editor_attempts": editor_attempts + 1,
        "status":          status_msg,
        "messages": [{"role": "assistant", "content": status_msg}]
    }