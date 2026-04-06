"""
Scholar Lens - Writer Agent

Responsibilities:
  1. Read current_section_index to know which section to write
  2. Pull scraped content from search_results for that section
  3. Write the section in academic style using research_angle + key_themes
  4. Append new SectionContent to state["sections"]
  5. If retrying (attempts > 1), read evaluator feedback and improve

Reads from state  → current_section_index, outline, search_results,
                    research_angle, key_themes, topic, sections
Writes to state   → sections (appends or updates current section)
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import ResearchState, SectionContent
import os


# ─────────────────────────────────────────────────────────
# STRUCTURED OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────

class WrittenSection(BaseModel):
    """Structured output for a single written section."""

    title: str = Field(
        description="The section title exactly as given."
    )
    content: str = Field(
        description="""Full written content for this section in academic style.
        Must be 400-700 words. Use formal academic language.
        Reference sources naturally within the text.
        Do not use bullet points — write in flowing paragraphs.
        Each paragraph should build on the previous one."""
    )
    citations: list[str] = Field(
        description="""List of raw citation strings extracted from the sources used.
        Format each as: 'Author/Title | URL | Year if available'
        These will be formatted properly by the Citation Agent later.
        Include 2-5 citations minimum."""
    )
    word_count: int = Field(
        description="Approximate word count of the content written."
    )


# ─────────────────────────────────────────────────────────
# WRITER NODE
# ─────────────────────────────────────────────────────────

def writer_node(state: ResearchState) -> dict:
    """
    Writes one section of the research paper.

    Two modes:
    - First attempt  → write fresh from search results
    - Retry attempt  → read evaluator feedback, rewrite and improve

    After writing, appends SectionContent to state["sections"].
    Evaluator node runs next to score it.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,        # Balanced — creative but academically grounded
        api_key=os.getenv("OPENAI_API_KEY")
    )

    structured_llm = llm.with_structured_output(WrittenSection)

    # ── What section are we writing? ────────────────────
    current_idx   = state["current_section_index"]
    section_title = state["outline"][current_idx]

    # ── Is this a retry? ────────────────────────────────
    # sections list already has this index if evaluator rejected it
    existing_sections = state.get("sections", [])
    is_retry = current_idx < len(existing_sections)

    if is_retry:
        current_section  = existing_sections[current_idx]
        attempts         = current_section["attempts"] + 1
        previous_content = current_section["content"]
        previous_score   = current_section["score"]
        previous_feedback = current_section.get("feedback", "Improve depth and academic tone.")
    else:
        attempts          = 1
        previous_content  = ""
        previous_score    = 0.0
        previous_feedback = ""  

    # ── Pull scraped content for this section ───────────
    search_results  = state.get("search_results", {})
    section_sources = search_results.get(section_title, [])
    sources_text    = "\n\n---\n\n".join(section_sources) if section_sources else "No sources available."

    # ── Key themes to maintain consistency ──────────────
    key_themes = state.get("key_themes", [])
    themes_str = ", ".join(key_themes) if key_themes else "Not specified"

    # ── Build messages ───────────────────────────────────
    system_prompt = """You are an expert academic writer producing a high-quality 
    research paper section. Your writing must be:
    - Formal academic tone — no casual language
    - Evidence-based — reference the provided sources naturally
    - Flowing paragraphs — no bullet points or headers within the section
    - 400-700 words — substantial but focused
    - Consistent with the paper's research angle and key themes
    - Properly attributed — mention sources by name/title in text

    Think like a PhD researcher writing for peer review."""

    if is_retry:
        # Retry mode — give writer the feedback and ask it to improve
        user_prompt = f"""
        RESEARCH PAPER TOPIC: {state['topic']}
        RESEARCH ANGLE: {state['research_angle']}
        KEY THEMES TO MAINTAIN: {themes_str}
        CITATION STYLE: {state['citation_style']}

        SECTION TO REWRITE: {section_title}
        ATTEMPT NUMBER: {attempts}

        ── PREVIOUS ATTEMPT (score: {previous_score}/10) ──
        {previous_content}

        ── EVALUATOR FEEDBACK — YOU MUST ADDRESS EVERY POINT ──
        {previous_feedback}

        ── REWRITE INSTRUCTIONS ──
        Do NOT reuse the same sentences from previous attempt.
        Write a substantially different and improved version.
        Address every point in the evaluator feedback above.
        Increase depth, add more specific citations from sources.

        ── AVAILABLE SOURCES ──
        {sources_text}

        Please rewrite this section addressing the quality issues.
        """
    else:
        # First attempt — write fresh
        user_prompt = f"""
        RESEARCH PAPER TOPIC: {state['topic']}
        RESEARCH ANGLE: {state['research_angle']}
        KEY THEMES TO MAINTAIN: {themes_str}
        CITATION STYLE: {state['citation_style']}

        SECTION TO WRITE: {section_title}
        ATTEMPT NUMBER: {attempts}

        ── AVAILABLE SOURCES (use these, do not hallucinate) ──
        {sources_text}

        Write this section in formal academic style.
        Reference the sources naturally within the text.
        Maintain the research angle throughout.
        """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    # ── Invoke LLM ───────────────────────────────────────
    written: WrittenSection = structured_llm.invoke(messages)

    # ── Build SectionContent for state ──────────────────
    new_section: SectionContent = {
        "title":     written.title,
        "content":   written.content,
        "score":     0.0,       # evaluator fills this next
        "attempts":  attempts,
        "citations": written.citations,
        "feedback":  ""  
    }

    # ── Update sections list ─────────────────────────────
    updated_sections = list(existing_sections)

    if is_retry:
        # Replace existing section at current index
        updated_sections[current_idx] = new_section
    else:
        # Append new section
        updated_sections.append(new_section)

    status_msg = (
        f"✍️ {'Rewrote' if is_retry else 'Wrote'} section "
        f"**{section_title}** "
        f"({written.word_count} words, attempt {attempts})"
    )

    return {
        "sections": updated_sections,
        "status":   status_msg,
        "messages": [{"role": "assistant", "content": status_msg}]
    }