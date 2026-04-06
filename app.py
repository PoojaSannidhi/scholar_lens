"""
Scholar Lens - Gradio App
"""

from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import gradio as gr
from graph import graph

OUTPUT_DIR = "/tmp/scholar_lens_outputs"

# To this — Gradio can always serve from working directory
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;1,8..60,300&display=swap');

:root {
    --bg-primary:    #0f0f0f;
    --bg-secondary:  #1a1a1a;
    --bg-card:       #222222;
    --accent:        #c9a84c;
    --accent-soft:   #e8c97a;
    --text-primary:  #f0ede6;
    --text-secondary:#a09880;
    --border:        #333333;
}

body, .gradio-container {
    background-color: var(--bg-primary) !important;
    font-family: 'Source Serif 4', Georgia, serif !important;
    color: var(--text-primary) !important;
}

.scholar-header {
    text-align: center;
    padding: 40px 24px 28px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}

.scholar-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0;
}

.scholar-title span { color: var(--accent); font-style: italic; }

.scholar-subtitle {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-top: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

textarea, input[type="text"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'Source Serif 4', Georgia, serif !important;
    border-radius: 2px !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    outline: none !important;
}

select {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 2px !important;
}

label, .label-wrap span {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

.approve-btn {
    background: #2e7d32 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Source Serif 4', serif !important;
    font-size: 0.9rem !important;
}

.reject-btn {
    background: transparent !important;
    color: #ef5350 !important;
    border: 1px solid #ef5350 !important;
    border-radius: 2px !important;
    font-family: 'Source Serif 4', serif !important;
    font-size: 0.9rem !important;
}

.dl-btn {
    background: var(--bg-card) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 2px !important;
    font-family: 'Source Serif 4', serif !important;
    font-size: 0.9rem !important;
    padding: 10px !important;
    width: 100% !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
"""


# ─────────────────────────────────────────────────────────
# GRAPH HELPERS
# ─────────────────────────────────────────────────────────

def _get_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _is_graph_paused(thread_id: str) -> bool:
    """
    Check if graph is paused at interrupt_after=["planner_node"].

    After interrupt, state.next contains the routing function name
    not the target node — because planner uses add_conditional_edges
    with dispatch_searches (Send() API).

    So we check:
    1. thread_id exists in SQLite (graph has run)
    2. outline is populated (planner finished)
    3. search_results is empty (search hasn't started yet)
    """
    try:
        if not thread_id:
            return False
        state  = graph.get_state(_get_config(thread_id))
        values = state.values
        # Planner done = outline exists
        # Search not started = search_results empty
        outline        = values.get("outline", [])
        search_results = values.get("search_results", {})
        return bool(outline) and not bool(search_results)
    except Exception:
        return False


def _extract_content(msg) -> str:
    """Extract content from LangChain message object or dict."""
    if isinstance(msg, dict):
        return msg.get("content", "")
    elif hasattr(msg, "content"):
        return msg.content          # LangChain AIMessage object
    return ""


# ─────────────────────────────────────────────────────────
# STREAM HELPERS
# ─────────────────────────────────────────────────────────

def _stream_graph(config: dict, initial_state: dict = None):
    # Increase recursion limit — 10 sections × 3 retries each = ~60+ steps
    # Default of 25 is too low for our pipeline
    stream_config = {**config, "recursion_limit": 200}
    for event in graph.stream(initial_state, stream_config, stream_mode="values"):
        yield event


def _process_chunk(chunk: dict, chat_history: list, pdf_path: str, md_path: str):
    """
    Extract new messages from chunk.
    Only add messages that weren't in previous chunk — avoid duplicates.
    """
    messages = chunk.get("messages", [])
    if messages:
        # Only process the LAST message — it's the newest one
        # Previous messages are already in chat_history
        last_msg = messages[-1]
        content  = _extract_content(last_msg)
        if content and (
            not chat_history or
            chat_history[-1].get("content") != content
        ):
            chat_history = chat_history + [{"role": "assistant", "content": content}]

    if chunk.get("pdf_path"):
        pdf_path = chunk["pdf_path"]
    if chunk.get("markdown_path"):
        md_path = chunk["markdown_path"]

    return chat_history, pdf_path, md_path


# ─────────────────────────────────────────────────────────
# CORE HANDLERS
# ─────────────────────────────────────────────────────────

def handle_approve(chat_history, thread_id, pdf_path, md_path):
    """User clicked Approve — resume graph from checkpoint."""
    config = _get_config(thread_id)
    chat_history = chat_history + [{"role": "assistant", "content": "✅ Outline approved — starting research..."}]
    yield chat_history, thread_id, pdf_path, md_path, "🔍 Searching...", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    for chunk in _stream_graph(config):
        chat_history, pdf_path, md_path = _process_chunk(chunk, chat_history, pdf_path, md_path)
        yield (
            chat_history, thread_id, pdf_path, md_path,
            chunk.get("status", ""),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=pdf_path if pdf_path and os.path.isfile(pdf_path) else None, visible=bool(pdf_path and os.path.isfile(pdf_path))),
            gr.update(value=md_path  if md_path  and os.path.isfile(md_path)  else None, visible=bool(md_path  and os.path.isfile(md_path)))
        )


def handle_reject(chat_history, thread_id, pdf_path, md_path):
    """User clicked Reject — rewind to planner."""
    config = _get_config(thread_id)
    graph.update_state(config, {
        "outline": [], "research_angle": "",
        "key_themes": [], "suggested_sources": [],
    }, as_node="planner_node")

    chat_history = chat_history + [{"role": "assistant", "content": "🔄 Regenerating outline..."}]
    yield chat_history, thread_id, pdf_path, md_path, "📋 Replanning...", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    for chunk in _stream_graph(config):
        chat_history, pdf_path, md_path = _process_chunk(chunk, chat_history, pdf_path, md_path)
        paused = _is_graph_paused(thread_id)
        yield chat_history, thread_id, pdf_path, md_path, chunk.get("status", ""), gr.update(visible=paused), gr.update(visible=paused), gr.update(visible=False), gr.update(visible=False)


def handle_feedback(message, chat_history, thread_id, pdf_path, md_path):
    """User typed feedback — enrich topic and replan."""
    if not message.strip():
        yield chat_history, thread_id, pdf_path, md_path, "", gr.update(), gr.update(), gr.update(), gr.update()
        return

    config        = _get_config(thread_id)
    current_state = graph.get_state(config).values
    enriched      = f"{current_state.get('topic', '')}. User feedback: {message}"

    graph.update_state(config, {
        "topic": enriched, "outline": [],
        "research_angle": "", "key_themes": [], "suggested_sources": [],
    }, as_node="planner_node")

    chat_history = chat_history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": "📝 Replanning with your feedback..."}
    ]
    yield chat_history, thread_id, pdf_path, md_path, "📋 Replanning...", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    for chunk in _stream_graph(config):
        chat_history, pdf_path, md_path = _process_chunk(chunk, chat_history, pdf_path, md_path)
        paused = _is_graph_paused(thread_id)
        yield chat_history, thread_id, pdf_path, md_path, chunk.get("status", ""), gr.update(visible=paused), gr.update(visible=paused), gr.update(visible=False), gr.update(visible=False)


def handle_start(topic, citation_style, chat_history, thread_id, pdf_path, md_path):
    """User clicked Begin Research — start fresh graph."""
    if not topic.strip():
        yield chat_history, thread_id, pdf_path, md_path, "Please enter a topic.", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        return

    thread_id    = str(uuid.uuid4())
    config       = _get_config(thread_id)
    chat_history = chat_history + [{"role": "assistant", "content": f"🎓 Starting research on: **{topic}**"}]

    yield chat_history, thread_id, pdf_path, md_path, "📋 Planning...", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    initial_state = {
        "topic":                 topic,
        "citation_style":        citation_style,
        "outline":               [],
        "research_angle":        "",
        "key_themes":            [],
        "suggested_sources":     [],
        "search_results":        {},
        "sections":              [],
        "current_section_index": 0,
        "abstract":              "",
        "final_paper":           "",
        "overall_score":         0.0,
        "editor_attempts":       0,
        "formatted_citations":   [],
        "pdf_path":              None,
        "markdown_path":         None,
        "messages":              [],
        "status":                "Starting..."
    }

    for chunk in _stream_graph(config, initial_state):
        chat_history, pdf_path, md_path = _process_chunk(chunk, chat_history, pdf_path, md_path)
        paused = _is_graph_paused(thread_id)
        yield (
            chat_history, thread_id, pdf_path, md_path,
            chunk.get("status", ""),
            gr.update(visible=paused),   # approve btn
            gr.update(visible=paused),   # reject btn
            gr.update(visible=bool(pdf_path)),
            gr.update(visible=bool(md_path))
        )


# ─────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="Scholar Lens",
        theme=gr.themes.Base(primary_hue="amber", neutral_hue="stone")
    ) as demo:

        # State
        thread_id_state = gr.State("")
        pdf_path_state  = gr.State("")
        md_path_state   = gr.State("")

        # Header
        gr.HTML("""
        <div class="scholar-header">
            <h1 class="scholar-title">Scholar <span>Lens</span></h1>
            <p class="scholar-subtitle">Multi-Agent Research Paper Generator &nbsp;·&nbsp; Powered by LangGraph</p>
        </div>
        """)

        with gr.Row():

            # ── LEFT COLUMN ──────────────────────────────
            with gr.Column(scale=1, min_width=300):

                topic_input = gr.Textbox(
                    label="Research Topic",
                    placeholder="e.g. Impact of Large Language Models on Software Engineering",
                    lines=3
                )

                citation_dropdown = gr.Dropdown(
                    choices=["APA", "IEEE", "MLA"],
                    value="APA",
                    label="Citation Style"
                )

                start_btn = gr.Button("✦ Begin Research", variant="primary")

                # Approve / Reject — hidden until outline is ready
                with gr.Row():
                    approve_btn = gr.Button(
                        "✅ Approve Outline",
                        variant="primary",
                        visible=False,
                        elem_classes=["approve-btn"]
                    )
                    reject_btn = gr.Button(
                        "✗ Reject",
                        variant="secondary",
                        visible=False,
                        elem_classes=["reject-btn"]
                    )

                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )

                # Feedback input — for when user wants to modify outline
                feedback_input = gr.Textbox(
                    label="Feedback on Outline",
                    placeholder="e.g. Add a section on ethical implications...",
                    lines=2,
                    visible=False
                )
                feedback_btn = gr.Button(
                    "Send Feedback",
                    variant="secondary",
                    visible=False
                )

                # Downloads
                gr.HTML("<p style='color:#a09880; font-size:0.8rem; letter-spacing:0.06em; text-transform:uppercase; margin-top:16px; margin-bottom:8px;'>Downloads</p>")

                pdf_download = gr.DownloadButton(
                    label="📄 Download PDF",
                    visible=False,
                    elem_classes=["dl-btn"]
                )
                md_download = gr.DownloadButton(
                    label="📝 Download Markdown",
                    visible=False,
                    elem_classes=["dl-btn"]
                )

            # ── RIGHT COLUMN — Chat ───────────────────────
            with gr.Column(scale=2):

                chatbot = gr.Chatbot(
                    height=620,
                    show_label=False,
                    elem_classes=["chatbot"]
                )

        # ── Outputs list ─────────────────────────────────
        # approve_btn, reject_btn, feedback_input, feedback_btn visibility
        # are controlled by whether graph is paused
        outputs = [
            chatbot, thread_id_state, pdf_path_state, md_path_state,
            status_text,
            approve_btn, reject_btn,       # shown when paused
            pdf_download, md_download      # shown when paper ready
        ]

        # ── Show feedback controls when paused ───────────
        approve_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=True)),
            outputs=[feedback_input, feedback_btn]
        )

        # ── Wire buttons ─────────────────────────────────
        start_btn.click(
            fn=handle_start,
            inputs=[topic_input, citation_dropdown, chatbot,
                    thread_id_state, pdf_path_state, md_path_state],
            outputs=outputs
        )

        approve_btn.click(
            fn=handle_approve,
            inputs=[chatbot, thread_id_state, pdf_path_state, md_path_state],
            outputs=outputs
        )

        reject_btn.click(
            fn=handle_reject,
            inputs=[chatbot, thread_id_state, pdf_path_state, md_path_state],
            outputs=outputs
        )

        feedback_btn.click(
            fn=handle_feedback,
            inputs=[feedback_input, chatbot, thread_id_state,
                    pdf_path_state, md_path_state],
            outputs=outputs
        ).then(fn=lambda: "", outputs=[feedback_input])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()