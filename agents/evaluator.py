"""
ResearchCraft - Evaluator Agent
Scores each section 1-10 using structured output (Pydantic).
This is the quality gate — sections below 7 loop back to the writer.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from state import ResearchState, SectionContent
import os


# ─────────────────────────────────────────────────────────
# STRUCTURED OUTPUT SCHEMA
# LangChain's .with_structured_output() forces the LLM
# to return exactly this shape — no hallucinated keys,
# no missing fields, fully typed and validated.
# ─────────────────────────────────────────────────────────

class SectionEvaluation(BaseModel):
    """Structured evaluation of a single research paper section."""

    score: float = Field(
        description="Overall quality score from 1.0 to 10.0",
        ge=1.0,
        le=10.0
    )
    clarity_score: float = Field(
        description="How clear and readable the writing is. 1-10",
        ge=1.0,
        le=10.0
    )
    depth_score: float = Field(
        description="How deep and thorough the academic analysis is. 1-10",
        ge=1.0,
        le=10.0
    )
    academic_tone_score: float = Field(
        description="How well it matches formal academic writing style. 1-10",
        ge=1.0,
        le=10.0
    )
    citation_usage_score: float = Field(
        description="How well citations and references are used. 1-10",
        ge=1.0,
        le=10.0
    )
    feedback: str = Field(
        description="Specific, actionable feedback for the writer to improve this section."
    )
    strengths: list[str] = Field(
        description="List of 2-3 specific things done well in this section."
    )
    improvements: list[str] = Field(
        description="List of 2-3 specific things that must be improved."
    )
    approved: bool = Field(
        description="True if score >= 7.0 and section is good enough to proceed."
    )


class FinalPaperEvaluation(BaseModel):
    """Structured evaluation of the complete assembled research paper."""

    overall_score: float = Field(
        description="Overall paper quality score from 1.0 to 10.0",
        ge=1.0,
        le=10.0
    )
    coherence_score: float = Field(
        description="How well all sections flow together as one paper. 1-10",
        ge=1.0,
        le=10.0
    )
    argument_score: float = Field(
        description="How strong and consistent the central argument/thesis is. 1-10",
        ge=1.0,
        le=10.0
    )
    citation_consistency_score: float = Field(
        description="How consistently citations are formatted throughout. 1-10",
        ge=1.0,
        le=10.0
    )
    abstract_alignment_score: float = Field(
        description="How well the abstract reflects the actual paper content. 1-10",
        ge=1.0,
        le=10.0
    )
    feedback: str = Field(
        description="Overall feedback for the final editor to improve the paper."
    )
    approved: bool = Field(
        description="True if overall_score >= 8.0 and paper is ready to export."
    )


# ─────────────────────────────────────────────────────────
# SECTION EVALUATOR NODE
# ─────────────────────────────────────────────────────────

def evaluator_node(state: ResearchState) -> dict:
    """
    Evaluates the current section using structured output.
    Updates the section's score and feedback in state.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,               # Deterministic scoring
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Bind structured output schema to LLM
    structured_llm = llm.with_structured_output(SectionEvaluation)

    current_idx = state["current_section_index"]
    current_section = state["sections"][current_idx]

    messages = [
        SystemMessage(content="""You are a strict academic peer reviewer evaluating 
        sections of a research paper. Be honest and critical. A score of 7+ means 
        the section is genuinely good — don't be generous. Score 1-10 on each dimension.
        Your feedback must be specific and actionable, not generic."""),

        HumanMessage(content=f"""
        Topic: {state['topic']}
        Research Angle: {state['research_angle']}
        Section Title: {current_section['title']}
        Attempt Number: {current_section['attempts']}

        === SECTION CONTENT ===
        {current_section['content']}

        === PREVIOUS FEEDBACK (if retry) ===
        {current_section.get('feedback', 'First attempt — no previous feedback')}
        Note if the writer has addressed the previous feedback.

        Please evaluate this section rigorously.
        """)
    ]

    evaluation: SectionEvaluation = structured_llm.invoke(messages)

    # Update the section with score AND feedback for writer to use on retry
    updated_sections = list(state["sections"])
    improvements_text = "\n".join(f"- {imp}" for imp in evaluation.improvements)
    updated_sections[current_idx] = {
        **current_section,
        "score":    evaluation.score,
        "feedback": f"{evaluation.feedback}\n\nSpecific improvements needed:\n{improvements_text}",
    }

    status_msg = (
        f"✅ Section '{current_section['title']}' approved (score: {evaluation.score}/10)"
        if evaluation.approved
        else f"🔄 Section '{current_section['title']}' needs revision (score: {evaluation.score}/10) — {evaluation.feedback[:100]}..."
    )

    return {
        "sections": updated_sections,
        "status": status_msg,
        "messages": [{"role": "assistant", "content": status_msg}]
    }


# ─────────────────────────────────────────────────────────
# FINAL PAPER EVALUATOR NODE
# ─────────────────────────────────────────────────────────

def final_evaluator_node(state: ResearchState) -> dict:
    """
    Evaluates the complete assembled paper.
    Checks coherence, flow, citation consistency, abstract alignment.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    structured_llm = llm.with_structured_output(FinalPaperEvaluation)

    messages = [
        SystemMessage(content="""You are a senior academic editor doing a final review 
        of a complete research paper before publication. Evaluate the paper as a whole — 
        not individual sections. Focus on coherence, flow, argument consistency, 
        and whether the abstract accurately reflects the paper. Be strict: 8+ means 
        genuinely publication-ready."""),

        HumanMessage(content=f"""
        Topic: {state['topic']}
        Citation Style: {state['citation_style']}

        === FULL PAPER ===
        {state['final_paper']}

        Please evaluate this complete research paper rigorously.
        """)
    ]

    evaluation: FinalPaperEvaluation = structured_llm.invoke(messages)

    status_msg = (
        f"✅ Paper approved for export (overall score: {evaluation.overall_score}/10)"
        if evaluation.approved
        else f"🔄 Paper needs final polish (score: {evaluation.overall_score}/10) — {evaluation.feedback[:100]}..."
    )

    return {
        "overall_score": evaluation.overall_score,
        "editor_attempts": state.get("editor_attempts", 0) + 1,
        "status": status_msg,
        "messages": [{"role": "assistant", "content": status_msg}]
    }