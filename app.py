"""
Scholar Lens - Gradio App
"""

from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import gradio as gr
from graph import graph

# Gradio 5 uses type="messages", Gradio 6 removed this parameter
GRADIO_V5 = int(gr.__version__.split(".")[0]) < 6
# BrowserState persists across reloads — only in Gradio 5+
# Fall back to regular State if not available
HAS_BROWSER_STATE = hasattr(gr, "BrowserState")


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
    color: #f0ede6;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.1;
}

.scholar-title span { color: #c9a84c; font-style: italic; }

.scholar-subtitle {
    font-size: 0.85rem;
    color: #a09880;
    margin-top: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

textarea, input[type="text"], input[type="search"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'Source Serif 4', Georgia, serif !important;
    border-radius: 2px !important;
}

textarea:focus, input:focus {
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
}

.reject-btn {
    background: transparent !important;
    color: #ef5350 !important;
    border: 1px solid #ef5350 !important;
    border-radius: 2px !important;
}

.continue-btn {
    background: var(--accent) !important;
    color: #0f0f0f !important;
    border: none !important;
    border-radius: 2px !important;
    font-weight: 600 !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
"""


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def _get_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _is_graph_paused(thread_id: str) -> bool:
    try:
        state = graph.get_state(_get_config(thread_id))
        values = state.values
        return bool(values.get("outline")) and not bool(values.get("search_results"))
    except Exception:
        return False


def _extract_content(msg) -> str:
    if isinstance(msg, dict):
        return msg.get("content", "")
    elif hasattr(msg, "content"):
        return msg.content
    return ""


def _stream_graph(config: dict, initial_state: dict = None):
    stream_config = {**config, "recursion_limit": 200}
    for event in graph.stream(initial_state, stream_config, stream_mode="values"):
        yield event


def _process_chunk(chunk: dict, chat: list, pdf: str, md: str):
    messages = chunk.get("messages", [])
    if messages:
        content = _extract_content(messages[-1])
        last_content = chat[-1].get("content", "") if chat else ""
        if content and content != last_content:
            chat = chat + [{"role": "assistant", "content": content}]

    new_pdf = chunk.get("pdf_path") or pdf
    new_md  = chunk.get("markdown_path") or md
    return chat, new_pdf, new_md


def _safe_file(path: str):
    """Return path only if it's a real file — prevents IsADirectoryError."""
    if path and os.path.isfile(path):
        return path
    return None


def _out(chat, thread, pdf, md, status,
         approve, reject, pdf_dl, md_dl, cont):
    """Helper to always return exactly 10 outputs."""
    return (chat, thread, pdf, md, status,
            approve, reject, pdf_dl, md_dl, cont)


# ─────────────────────────────────────────────────────────
# HANDLERS
# ─────────────────────────────────────────────────────────

def handle_start(topic, citation, chat, thread, pdf, md):
    if not topic.strip():
        yield _out(chat, thread, pdf, md, "Please enter a topic.",
                   gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        return

    thread = str(uuid.uuid4())
    config = _get_config(thread)
    chat   = chat + [{"role": "assistant", "content": f"🎓 Starting: **{topic}**"}]

    yield _out(chat, thread, "", "", "📋 Planning...",
               gr.update(visible=False), gr.update(visible=False),
               gr.update(visible=False), gr.update(visible=False),
               gr.update(visible=False))

    initial_state = {
        "topic":                 topic,
        "citation_style":        citation,
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
        chat, pdf, md = _process_chunk(chunk, chat, pdf, md)
        paused = _is_graph_paused(thread)
        sf_pdf = _safe_file(pdf)
        sf_md  = _safe_file(md)

        yield _out(chat, thread, pdf, md,
                   chunk.get("status", ""),
                   gr.update(visible=paused),
                   gr.update(visible=paused),
                   gr.update(value=sf_pdf, visible=bool(sf_pdf)),
                   gr.update(value=sf_md,  visible=bool(sf_md)),
                   gr.update(visible=False))


def handle_approve(chat, thread, pdf, md):
    config = _get_config(thread)
    chat   = chat + [{"role": "assistant", "content": "✅ Approved — starting research..."}]

    yield _out(chat, thread, pdf, md, "🔍 Searching...",
               gr.update(visible=False), gr.update(visible=False),
               gr.update(visible=False), gr.update(visible=False),
               gr.update(visible=False))

    for chunk in _stream_graph(config):
        chat, pdf, md = _process_chunk(chunk, chat, pdf, md)
        sf_pdf = _safe_file(pdf)
        sf_md  = _safe_file(md)

        yield _out(chat, thread, pdf, md,
                   chunk.get("status", ""),
                   gr.update(visible=False),
                   gr.update(visible=False),
                   gr.update(value=sf_pdf, visible=bool(sf_pdf)),
                   gr.update(value=sf_md,  visible=bool(sf_md)),
                   gr.update(visible=False))


def handle_reject(chat, thread, pdf, md):
    config = _get_config(thread)
    graph.update_state(config, {
        "outline": [], "research_angle": "",
        "key_themes": [], "suggested_sources": [],
    }, as_node="planner_node")

    chat = chat + [{"role": "assistant", "content": "🔄 Regenerating outline..."}]

    yield _out(chat, thread, pdf, md, "📋 Replanning...",
               gr.update(visible=False), gr.update(visible=False),
               gr.update(visible=False), gr.update(visible=False),
               gr.update(visible=False))

    for chunk in _stream_graph(config):
        chat, pdf, md = _process_chunk(chunk, chat, pdf, md)
        paused = _is_graph_paused(thread)

        yield _out(chat, thread, pdf, md,
                   chunk.get("status", ""),
                   gr.update(visible=paused),
                   gr.update(visible=paused),
                   gr.update(visible=False),
                   gr.update(visible=False),
                   gr.update(visible=False))


def handle_feedback(feedback, chat, thread, pdf, md):
    if not feedback.strip():
        yield _out(chat, thread, pdf, md, "",
                   gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        return

    config = _get_config(thread)
    current = graph.get_state(config).values
    enriched = f"{current.get('topic', '')}. User feedback: {feedback}"
    graph.update_state(config, {
        "topic": enriched, "outline": [],
        "research_angle": "", "key_themes": [], "suggested_sources": [],
    }, as_node="planner_node")

    chat = chat + [
        {"role": "user",      "content": feedback},
        {"role": "assistant", "content": "📝 Replanning with your feedback..."}
    ]

    yield _out(chat, thread, pdf, md, "📋 Replanning...",
               gr.update(visible=False), gr.update(visible=False),
               gr.update(visible=False), gr.update(visible=False),
               gr.update(visible=False))

    for chunk in _stream_graph(config):
        chat, pdf, md = _process_chunk(chunk, chat, pdf, md)
        paused = _is_graph_paused(thread)

        yield _out(chat, thread, pdf, md,
                   chunk.get("status", ""),
                   gr.update(visible=paused),
                   gr.update(visible=paused),
                   gr.update(visible=False),
                   gr.update(visible=False),
                   gr.update(visible=False))


def handle_continue(chat, thread, pdf, md):
    config = _get_config(thread)
    chat   = chat + [{"role": "assistant", "content": "▶ Continuing research..."}]

    yield _out(chat, thread, pdf, md, "▶ Resuming...",
               gr.update(visible=False), gr.update(visible=False),
               gr.update(visible=False), gr.update(visible=False),
               gr.update(visible=False))

    for chunk in _stream_graph(config):
        chat, pdf, md = _process_chunk(chunk, chat, pdf, md)
        sf_pdf = _safe_file(pdf)
        sf_md  = _safe_file(md)

        yield _out(chat, thread, pdf, md,
                   chunk.get("status", ""),
                   gr.update(visible=False),
                   gr.update(visible=False),
                   gr.update(value=sf_pdf, visible=bool(sf_pdf)),
                   gr.update(value=sf_md,  visible=bool(sf_md)),
                   gr.update(visible=False))


# ─────────────────────────────────────────────────────────
# BUILD UI
# ─────────────────────────────────────────────────────────

def build_ui():
    blocks_kwargs = dict(title="Scholar Lens")
    if GRADIO_V5:
        blocks_kwargs["css"]   = CUSTOM_CSS
        blocks_kwargs["theme"] = gr.themes.Base(primary_hue="amber", neutral_hue="stone")
    with gr.Blocks(**blocks_kwargs) as demo:

        # ── State ────────────────────────────────────────
        # BrowserState persists thread_id in browser localStorage
        # Falls back to gr.State if not available in this Gradio version
        thread = gr.BrowserState("") if HAS_BROWSER_STATE else gr.State("")
        pdf    = gr.State("")
        md     = gr.State("")

        # ── Header ──────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center;padding:40px 24px 28px;border-bottom:1px solid #333;margin-bottom:28px;">
            <h1 style="font-family:'Playfair Display',Georgia,serif;font-size:2.8rem;font-weight:700;color:#f0ede6 !important;letter-spacing:-0.02em;margin:0;">
                Scholar <span style="color:#c9a84c;font-style:italic;">Lens</span>
            </h1>
            <p style="font-size:0.85rem;color:#a09880;margin-top:10px;letter-spacing:0.08em;text-transform:uppercase;">
                Multi-Agent Research Paper Generator &nbsp;·&nbsp; Powered by LangGraph
            </p>
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

                start_btn   = gr.Button("✦ Begin Research", variant="primary")

                with gr.Row():
                    approve_btn = gr.Button("✅ Approve Outline", variant="primary",
                                            visible=False, elem_classes=["approve-btn"])
                    reject_btn  = gr.Button("✗ Reject", variant="secondary",
                                            visible=False, elem_classes=["reject-btn"])

                continue_btn = gr.Button("▶ Continue Research", variant="secondary",
                                         visible=False, elem_classes=["continue-btn"])

                status_text = gr.Textbox(label="Status", interactive=False, lines=2)

                feedback_input = gr.Textbox(
                    label="Feedback on Outline",
                    placeholder="e.g. Add a section on ethical implications...",
                    lines=2, visible=False
                )
                feedback_btn = gr.Button("Send Feedback", variant="secondary", visible=False)

                gr.HTML("<p style='color:#a09880;font-size:0.8rem;letter-spacing:0.06em;text-transform:uppercase;margin-top:16px;margin-bottom:8px;'>Downloads</p>")

                pdf_download = gr.File(label="📄 PDF",      visible=False, interactive=False)
                md_download  = gr.File(label="📝 Markdown", visible=False, interactive=False)

            # ── RIGHT COLUMN ─────────────────────────────
            with gr.Column(scale=2):
                chatbot_kwargs = dict(
                    height=620,
                    show_label=False,
                    elem_classes=["chatbot"]
                )
                if GRADIO_V5:
                    chatbot_kwargs["type"] = "messages"
                chatbot = gr.Chatbot(**chatbot_kwargs)

        # ── Outputs ──────────────────────────────────────
        outputs = [chatbot, thread, pdf, md, status_text,
                   approve_btn, reject_btn, pdf_download, md_download, continue_btn]

        # ── On load — restore session ────────────────────
        def on_load(thread_id):
            empty = ([], "", "", "",
                     gr.update(visible=False), gr.update(visible=False),
                     gr.update(visible=False), gr.update(visible=False),
                     gr.update(visible=False))
            if not thread_id:
                return empty
            try:
                snap = graph.get_state(_get_config(thread_id))
                if not snap or not snap.values:
                    return empty
                values   = snap.values
                topic    = values.get("topic", "")
                sections = values.get("sections", [])
                if not topic:
                    return empty

                r_pdf = _safe_file(values.get("pdf_path"))
                r_md  = _safe_file(values.get("markdown_path"))

                history = [{"role": "assistant",
                            "content": "🔄 Resuming: **" + topic + "**"}]
                for s in sections:
                    title = s.get("title", "Unknown")
                    score = s.get("score", 0.0)
                    history.append({"role": "assistant",
                                    "content": "✅ '" + title + "' — " + str(score) + "/10"})
                if r_pdf:
                    history.append({"role": "assistant",
                                    "content": "🎉 Paper complete — downloads restored below."})

                paused    = _is_graph_paused(thread_id)
                is_mid    = bool(sections) and not paused and not r_pdf

                if paused:
                    outline = values.get("outline", [])
                    ol_text = "\n".join(str(i+1) + ". " + s for i, s in enumerate(outline))
                    history.append({"role": "assistant",
                                    "content": "⏸️ Outline ready:\n" + ol_text + "\n\nClick Approve to continue."})
                if is_mid:
                    history.append({"role": "assistant",
                                    "content": "⏯️ Generation in progress — click Continue Research."})

                return (
                    history,
                    "Resumed: " + topic,
                    r_pdf or "", r_md or "",
                    gr.update(visible=paused),
                    gr.update(visible=paused),
                    gr.update(value=r_pdf, visible=bool(r_pdf)),
                    gr.update(value=r_md,  visible=bool(r_md)),
                    gr.update(visible=is_mid)
                )
            except Exception as e:
                print(f"on_load error: {e}")
                import traceback
                traceback.print_exc()
                return empty

        demo.load(fn=on_load, inputs=[thread],
                  outputs=[chatbot, status_text, pdf, md,
                           approve_btn, reject_btn,
                           pdf_download, md_download, continue_btn])

        # ── Show feedback on approve click ───────────────
        approve_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=True)),
            outputs=[feedback_input, feedback_btn]
        )

        # ── Wire all buttons ─────────────────────────────
        start_btn.click(
            fn=handle_start,
            inputs=[topic_input, citation_dropdown, chatbot, thread, pdf, md],
            outputs=outputs
        )
        approve_btn.click(
            fn=handle_approve,
            inputs=[chatbot, thread, pdf, md],
            outputs=outputs
        )
        reject_btn.click(
            fn=handle_reject,
            inputs=[chatbot, thread, pdf, md],
            outputs=outputs
        )
        feedback_btn.click(
            fn=handle_feedback,
            inputs=[feedback_input, chatbot, thread, pdf, md],
            outputs=outputs
        ).then(fn=lambda: "", outputs=[feedback_input])

        continue_btn.click(
            fn=handle_continue,
            inputs=[chatbot, thread, pdf, md],
            outputs=outputs
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    launch_kwargs = dict(server_name="0.0.0.0", server_port=7860)
    if not GRADIO_V5:
        launch_kwargs["css"] = CUSTOM_CSS
    demo.launch(**launch_kwargs)