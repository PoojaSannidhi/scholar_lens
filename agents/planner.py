"""
ResearchCraft (Scholar Lens) - Planner Agent

First node in the graph.
Responsibilities:
  1. Understand the topic deeply
  2. Decide a unique research angle / thesis
  3. Generate a structured outline (list of sections)

After this node, graph PAUSES (interrupt_after=["planner_node"])
so user can review and edit the outline before research begins.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import ResearchState
import os


# ─────────────────────────────────────────────────────────
# STRUCTURED OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────

class PaperOutline(BaseModel):
    """Structured plan for the research paper."""

    research_angle: str = Field(
        description="""A unique, specific thesis or angle for this paper. 
        Not just the topic restated — a genuine intellectual position or 
        framing that makes this paper's argument clear and distinctive.
        Example: Instead of 'AI in healthcare', write 
        'LLMs as diagnostic assistants reduce time-to-diagnosis but 
        introduce new liability gaps in clinical workflows.'"""
    )

    section_titles: list[str] = Field(
        description="""Ordered list of section titles for the paper.
        Must always start with 'Introduction' and end with 'Conclusion'.
        Include 'Literature Review' and 'Methodology' where appropriate.
        Typically 5-7 sections total. Each title should be specific to 
        the topic, not generic. 
        Example: ['Introduction', 'Literature Review', 
        'LLMs in Clinical Diagnosis', 'Liability and Ethical Gaps',
        'Case Studies', 'Discussion', 'Conclusion']"""
    )

    estimated_word_count: int = Field(
        description="Estimated total word count for the full paper. Typically 3000-6000 words."
    )

    key_themes: list[str] = Field(
        description="3-5 key themes or concepts that should appear consistently throughout the paper."
    )

    suggested_sources: list[str] = Field(
        description="""3-5 suggested search queries to find relevant academic sources.
        These will be passed to the Search Agent.
        Example: ['LLMs clinical diagnosis accuracy 2024', 
        'medical AI liability legal framework',
        'GPT-4 radiology diagnostic studies arxiv']"""
    )


# ─────────────────────────────────────────────────────────
# PLANNER NODE
# ─────────────────────────────────────────────────────────

def planner_node(state: ResearchState) -> dict:
    """
    Takes user topic and generates a structured paper outline.
    Graph pauses after this node for human-in-the-loop approval.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,        # Some creativity for unique research angle
        api_key=os.getenv("OPENAI_API_KEY")
    )

    structured_llm = llm.with_structured_output(PaperOutline)

    messages = [
        SystemMessage(content="""You are an expert academic research planner with deep 
        knowledge across all domains. Your job is to take a research topic and create 
        a compelling, well-structured paper outline.

        Key responsibilities:
        - Find a UNIQUE angle — don't just restate the topic
        - Structure sections so they build on each other logically  
        - Make section titles specific, not generic
        - Suggest search queries that will find real academic sources
        - Think like a PhD supervisor reviewing a student's paper plan

        The outline will be shown to the user for approval before any writing begins.
        Make it impressive enough that the user says yes immediately."""),

        HumanMessage(content=f"""
        Research Topic: {state['topic']}
        Citation Style: {state['citation_style']}

        Please create a detailed, compelling outline for this research paper.
        Find a unique angle that makes this paper stand out from generic treatments
        of the topic.
        """)
    ]

    # Invoke structured LLM — returns PaperOutline Pydantic object
    plan: PaperOutline = structured_llm.invoke(messages)

    # Format outline for display in Gradio UI
    outline_display = _format_outline_for_display(plan)

    status_msg = f"📋 Outline ready — {len(plan.section_titles)} sections planned. Please review before research begins."

    return {
        # Core planner outputs written to state
        "outline": plan.section_titles,           # List[str] → outline in state
        "research_angle": plan.research_angle,
        "key_themes": plan.key_themes,            # ← NEW: writer reads for consistency
        "suggested_sources": plan.suggested_sources,  # ← NEW: search agent reads

        # UI updates
        "status": status_msg,
        "messages": [
            {
                "role": "assistant",
                "content": outline_display
            }
        ]
    }


# ─────────────────────────────────────────────────────────
# HELPER — Format Outline for Gradio Display
# ─────────────────────────────────────────────────────────

def _format_outline_for_display(plan: PaperOutline) -> str:
    """
    Converts PaperOutline into a readable markdown string
    that streams to the Gradio chatbox for user review.
    """

    sections_numbered = "\n".join(
        [f"  {i+1}. {section}" for i, section in enumerate(plan.section_titles)]
    )

    themes_list = "\n".join(
        [f"  • {theme}" for theme in plan.key_themes]
    )

    return f"""
## 📄 Paper Outline Ready for Your Review

**Research Angle:**
> {plan.research_angle}

**Sections ({len(plan.section_titles)} total, ~{plan.estimated_word_count:,} words):**
{sections_numbered}

**Key Themes Running Throughout:**
{themes_list}

**Estimated Word Count:** {plan.estimated_word_count:,} words

---
⏸️ **Graph is paused.** 
Type **"approve"** to start research, or tell me what to change in the outline.
"""