"""
Scholar Lens - Abstract Writer Agent

Written LAST — after all sections and citations are complete.
This mirrors real academic writing practice:
abstract summarizes what was actually written, not what was planned.

Reads from state  → sections, research_angle, key_themes, 
                    topic, citation_style, formatted_citations
Writes to state   → abstract: str

Uses GPT-4o-mini — synthesis and precision task.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import ResearchState
import os


# ─────────────────────────────────────────────────────────
# STRUCTURED OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────

class WrittenAbstract(BaseModel):
    """Structured output for the paper abstract."""

    abstract: str = Field(
        description="""A complete academic abstract of 150-250 words.
        Must follow the standard abstract structure:
        1. Context/Background  — why this topic matters
        2. Problem Statement   — what gap or question this paper addresses
        3. Methodology         — how the paper approaches the problem
        4. Key Findings        — what the paper concludes
        5. Implications        — why it matters to the field

        Must be written in third person, past tense for findings.
        No citations in the abstract.
        No bullet points — one flowing paragraph."""
    )

    keywords: list[str] = Field(
        description="""5-8 academic keywords for this paper.
        These appear below the abstract in published papers.
        Should be specific enough for database indexing.
        Example: ['large language models', 'software engineering',
        'code generation', 'developer productivity']"""
    )

    word_count: int = Field(
        description="Word count of the abstract. Must be between 150-250."
    )


# ─────────────────────────────────────────────────────────
# ABSTRACT NODE
# ─────────────────────────────────────────────────────────

def abstract_node(state: ResearchState) -> dict:
    """
    Writes the abstract AFTER all sections are complete.

    Why last?
    - Abstract must reflect what was ACTUALLY written
    - Not what was planned — real content may differ from outline
    - This is standard academic practice
    - Prevents abstract/paper misalignment (Final Evaluator checks this)
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,        # Mostly precise, slight creativity for flow
        api_key=os.getenv("OPENAI_API_KEY")
    )

    structured_llm = llm.with_structured_output(WrittenAbstract)

    # ── Collect all section content for context ──────────
    # Abstract writer reads the ENTIRE paper before writing
    full_paper_content = "\n\n".join([
        f"=== {section['title'].upper()} ===\n{section['content']}"
        for section in state["sections"]
    ])

    key_themes = state.get("key_themes", [])
    themes_str = ", ".join(key_themes) if key_themes else "Not specified"

    messages = [
        SystemMessage(content="""You are an expert academic writer specializing 
        in writing abstracts. You have read the entire paper and must now 
        write a precise, compelling abstract that accurately reflects 
        the paper's actual content — not what was planned, but what was written.

        A great abstract:
        - Stands alone — reader understands the paper without reading it
        - Is accurate — every claim in abstract exists in the paper
        - Is concise — 150-250 words, no filler
        - Follows the 5-part structure: context, problem, method, findings, implications
        - Uses formal academic language throughout"""),

        HumanMessage(content=f"""
        Paper Topic: {state['topic']}
        Research Angle: {state['research_angle']}
        Key Themes: {themes_str}

        === FULL PAPER CONTENT ===
        {full_paper_content}

        Please write the abstract for this paper.
        Base it entirely on what was actually written above.
        Do not introduce new claims not present in the paper.
        """)
    ]

    result: WrittenAbstract = structured_llm.invoke(messages)

    # Format abstract with keywords for display
    keywords_str = ", ".join(result.keywords)
    abstract_with_keywords = (
        f"{result.abstract}\n\n"
        f"**Keywords:** {keywords_str}"
    )

    status_msg = (
        f"📝 Abstract written — "
        f"{result.word_count} words, "
        f"{len(result.keywords)} keywords."
    )

    return {
        "abstract": abstract_with_keywords,
        "status": status_msg,
        "messages": [{"role": "assistant", "content": status_msg}]
    }