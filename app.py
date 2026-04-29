import html
import os
import re
import socket
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from legal_ai.config import APP_TITLE, CLAUSE_QUERIES  # noqa: E402
from legal_ai.generation import ContractAssistant, get_generator, get_openai_client  # noqa: E402
from legal_ai.retrieval import ContractRetriever  # noqa: E402

load_dotenv()

CSS = """
:root {
  --brand: #164c8f;
  --brand-dark: #0f315d;
  --accent: #2aa198;
  --ink: #182233;
  --muted: #5c6677;
  --line: #d7dee9;
  --soft: #f5f7fb;
  --paper: #ffffff;
  --warn-bg: #fff2df;
  --warn-line: #d9822b;
  --danger-bg: #fff0f1;
  --danger-line: #c93c4a;
  --ok-bg: #edf8f4;
  --ok-line: #2a9d78;
}
.gradio-container {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background:
    linear-gradient(180deg, #eef3f8 0%, #f8fafc 42%, #ffffff 100%);
}
.app-shell {
  max-width: 1240px;
  margin: 0 auto;
}
.hero {
  padding: 18px 0 14px;
  border-bottom: 1px solid rgba(22, 76, 143, 0.16);
  margin-bottom: 16px;
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 20px;
}
.hero h1 {
  font-size: 38px;
  line-height: 1.1;
  margin: 0 0 8px;
  color: var(--brand-dark);
}
.hero p {
  color: var(--muted);
  font-size: 16px;
  margin: 0;
  max-width: 760px;
}
.hero-badge {
  border: 1px solid rgba(42, 161, 152, 0.35);
  background: #ecfbf8;
  color: #0e665f;
  padding: 8px 12px;
  font-weight: 700;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  white-space: nowrap;
}
.status-box textarea {
  font-size: 14px !important;
}
.control-strip {
  align-items: end;
}
.flyer-output {
  min-height: 680px;
}
.brief-shell,
.mini-brief {
  border: 1px solid var(--line);
  background: var(--paper);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 18px 44px rgba(22, 35, 55, 0.08);
}
.brief-header {
  background: linear-gradient(135deg, var(--brand-dark), var(--brand));
  color: white;
  padding: 26px 28px;
  display: flex;
  justify-content: space-between;
  gap: 18px;
}
.brief-kicker {
  margin: 0 0 8px;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #bfe5ff;
}
.brief-header h2 {
  margin: 0;
  font-size: 30px;
  line-height: 1.1;
  color: white;
}
.brief-subtitle {
  margin: 10px 0 0;
  color: #d8e8f7;
  max-width: 720px;
}
.brief-stamp {
  align-self: start;
  border: 1px solid rgba(255,255,255,0.42);
  border-radius: 999px;
  padding: 8px 12px;
  font-size: 12px;
  font-weight: 800;
  white-space: nowrap;
}
.brief-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 14px;
  padding: 18px;
}
.brief-card {
  border: 1px solid var(--line);
  border-left: 5px solid var(--brand);
  border-radius: 8px;
  background: #ffffff;
  padding: 16px;
}
.brief-card.warning {
  background: var(--warn-bg);
  border-color: #f0c693;
  border-left-color: var(--warn-line);
}
.brief-card.danger {
  background: var(--danger-bg);
  border-color: #efb4bb;
  border-left-color: var(--danger-line);
}
.brief-card.citation {
  background: #f6f8fb;
  border-left-color: #778397;
}
.brief-card.source {
  background: #fbfcfe;
  border-left-color: #58677d;
}
.brief-card.clarify {
  background: #f4f8ff;
  border-left-color: #577bc1;
}
.brief-card h3 {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 0 10px;
  font-size: 17px;
  color: var(--ink);
}
.icon-dot {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-weight: 900;
  color: white;
  background: var(--brand);
  flex: 0 0 auto;
}
.warning .icon-dot {
  background: var(--warn-line);
}
.danger .icon-dot {
  background: var(--danger-line);
}
.clarify .icon-dot {
  background: #577bc1;
}
.citation .icon-dot {
  background: #778397;
}
.source .icon-dot {
  background: #58677d;
}
.brief-card a {
  color: var(--brand);
  font-weight: 800;
  text-decoration: none;
  border-bottom: 1px solid rgba(22, 76, 143, 0.28);
}
.brief-card a:hover {
  border-bottom-color: var(--brand);
}
.source-list {
  display: grid;
  gap: 10px;
  margin-top: 6px;
}
.source-chip {
  border: 1px solid #dce3ee;
  border-radius: 8px;
  background: white;
  padding: 10px 12px;
}
.source-chip-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 6px;
}
.source-chip-label {
  color: var(--brand);
  font-weight: 900;
}
.source-chip-meta {
  color: var(--muted);
  font-size: 12px;
}
.source-chip p {
  margin: 0;
  font-size: 13px;
  line-height: 1.45;
}
.brief-card p {
  margin: 0 0 10px;
  color: #2c3546;
  line-height: 1.5;
}
.brief-card ul {
  margin: 8px 0 0 18px;
  padding: 0;
}
.brief-card li {
  margin: 6px 0;
}
.brief-card.full {
  grid-column: 1 / -1;
}
.clause-list {
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
  padding: 18px;
}
.clause-block {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
  overflow: hidden;
}
.clause-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 16px;
  background: #f2f6fb;
  border-bottom: 1px solid var(--line);
}
.clause-title h3 {
  margin: 0;
  color: var(--brand-dark);
  font-size: 19px;
}
.clause-label {
  border: 1px solid rgba(22, 76, 143, 0.22);
  border-radius: 999px;
  padding: 5px 9px;
  color: var(--brand);
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
}
.clause-sections {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  padding: 14px;
}
.clause-sections .brief-card {
  box-shadow: none;
}
.section-note {
  color: var(--muted);
  font-size: 13px;
  margin: 8px 18px 0;
}
.brief-footer {
  border-top: 1px solid var(--line);
  padding: 14px 18px;
  color: var(--muted);
  font-size: 13px;
  background: #fafbfd;
}
.qa-panel,
.upload-panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
  padding: 14px;
  box-shadow: 0 12px 28px rgba(22, 35, 55, 0.06);
}
.side-title {
  margin: 0 0 8px;
  color: var(--brand-dark);
  font-size: 18px;
}
.preview-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
  padding: 14px;
  max-height: 280px;
  overflow: auto;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.45;
}
.disclaimer {
  color: var(--warn);
  font-size: 13px;
}
button.primary {
  background: var(--brand) !important;
}
@media (max-width: 900px) {
  .hero {
    align-items: flex-start;
    flex-direction: column;
  }
  .hero-badge {
    white-space: normal;
    max-width: 100%;
  }
  .brief-header {
    flex-direction: column;
  }
  .brief-grid,
  .clause-sections {
    grid-template-columns: 1fr;
  }
  .brief-card.full {
    grid-column: auto;
  }
}
"""


runtime: tuple[ContractRetriever, ContractAssistant] | None = None


def build_runtime() -> tuple[ContractRetriever, ContractAssistant]:
    client = get_openai_client()
    retriever = ContractRetriever(client)
    assistant = ContractAssistant(retriever=retriever, generator=get_generator())
    return retriever, assistant


def get_runtime() -> tuple[ContractRetriever, ContractAssistant]:
    global runtime
    if runtime is None:
        runtime = build_runtime()
    return runtime


def upload_contract(file):
    if file is None:
        return "Upload a PDF, DOCX, or HTML contract to begin.", None, preview_html()

    try:
        retriever, _assistant = get_runtime()
        uploaded = retriever.index_upload(file.name)
        status = (
            f"Ready: {uploaded.source_document}\n"
            f"Indexed chunks: {uploaded.num_chunks}\n"
            f"Temporary collection: {uploaded.collection_name}"
        )
        state = {
            "collection_name": uploaded.collection_name,
            "source_document": uploaded.source_document,
        }
        preview = preview_html(uploaded.raw_text_preview, uploaded.source_document)
        return status, state, preview
    except Exception as exc:
        return f"Upload failed: {exc}", None, ""


def require_document(doc_state):
    if not doc_state:
        return render_empty_brief(
            "Upload Required",
            "Please upload and process a contract first.",
        )
    return None


def summarize_contract(doc_state):
    missing = require_document(doc_state)
    if missing:
        return missing
    _retriever, assistant = get_runtime()
    result = assistant.summarize(doc_state["collection_name"], doc_state["source_document"])
    return render_brief(
        "Contract Brief",
        doc_state["source_document"],
        result,
        stamp="Review",
    )


def extract_key_clauses(doc_state):
    missing = require_document(doc_state)
    if missing:
        return missing
    _retriever, assistant = get_runtime()
    result = assistant.key_clauses(doc_state["collection_name"], doc_state["source_document"])
    return render_clause_watchlist(doc_state["source_document"], result)


def ask_question(question, doc_state):
    missing = require_document(doc_state)
    if missing:
        return missing
    _retriever, assistant = get_runtime()
    result = assistant.answer_question(question, doc_state["collection_name"], doc_state["source_document"])
    return render_brief(
        "Question Answer",
        doc_state["source_document"],
        result,
        stamp="Q&A",
    )


def render_empty_brief(title: str, message: str) -> str:
    return f"""
    <div class="brief-shell flyer-output">
      <div class="brief-header">
        <div>
          <p class="brief-kicker">Contract Clarity</p>
          <h2>{html.escape(title)}</h2>
          <p class="brief-subtitle">{html.escape(message)}</p>
        </div>
        <div class="brief-stamp">Ready</div>
      </div>
      <div class="brief-grid">
        <div class="brief-card full">
          <h3><span class="icon-dot">i</span> What You’ll Get</h3>
          <ul>
            <li>A flyer-style plain-English contract summary.</li>
            <li>Warning callouts for clauses that deserve attention.</li>
            <li>Questions to clarify before signing.</li>
            <li>Citation-backed answers from the uploaded document.</li>
          </ul>
        </div>
      </div>
      <div class="brief-footer">This tool explains contract text. It does not provide legal advice.</div>
    </div>
    """


def preview_html(text: str = "", title: str = "No document uploaded yet") -> str:
    body = html.escape(text or "Upload a contract to see a short text preview here.")
    return f"""
    <div class="preview-card">
      <h3 class="side-title">{html.escape(title)}</h3>
      <p>{body}</p>
    </div>
    """


def render_brief(title: str, source_document: str, markdown_text: str, stamp: str) -> str:
    sections = parse_markdown_sections(markdown_text)
    cards = []
    for heading, body in sections:
        cards.append(render_section_card(heading, body))
    if not cards:
        cards.append(render_section_card("Output", markdown_text))

    return f"""
    <div class="brief-shell flyer-output">
      <div class="brief-header">
        <div>
          <p class="brief-kicker">Contract Clarity</p>
          <h2>{html.escape(title)}</h2>
          <p class="brief-subtitle">{html.escape(source_document)} reviewed in a plain-English, evidence-backed format.</p>
        </div>
        <div class="brief-stamp">{html.escape(stamp)}</div>
      </div>
      <div class="brief-grid">
        {''.join(cards)}
      </div>
      <div class="brief-footer">
        Generated from retrieved contract context. This is not legal advice and does not replace review by an attorney.
      </div>
    </div>
    """


def render_clause_watchlist(source_document: str, markdown_text: str) -> str:
    clause_blocks = split_clause_blocks(markdown_text)
    if not clause_blocks:
        return render_brief("Clause Watchlist", source_document, markdown_text, stamp="Clauses")

    rendered_clauses = []
    for clause_name, body in clause_blocks:
        sub_sections = parse_markdown_sections(body)
        cards = [render_section_card(heading, section_body) for heading, section_body in sub_sections]
        if not cards:
            cards = [render_section_card("Plain-English Meaning", body)]
        rendered_clauses.append(
            f"""
            <article class="clause-block">
              <div class="clause-title">
                <h3>{html.escape(clause_name)}</h3>
                <span class="clause-label">Clause</span>
              </div>
              <div class="clause-sections">
                {''.join(cards)}
              </div>
            </article>
            """
        )

    return f"""
    <div class="brief-shell flyer-output">
      <div class="brief-header">
        <div>
          <p class="brief-kicker">Contract Clarity</p>
          <h2>Clause Watchlist</h2>
          <p class="brief-subtitle">{html.escape(source_document)} organized by clause type so each issue is easy to scan.</p>
        </div>
        <div class="brief-stamp">Clauses</div>
      </div>
      <p class="section-note">Each clause below is separated into its own block. Start with the orange watch-outs, then check citations.</p>
      <div class="clause-list">
        {''.join(rendered_clauses)}
      </div>
      <div class="brief-footer">
        Generated from retrieved contract context. This is not legal advice and does not replace review by an attorney.
      </div>
    </div>
    """


def split_clause_blocks(markdown_text: str) -> list[tuple[str, str]]:
    clause_names = set(CLAUSE_QUERIES.keys())
    blocks: list[tuple[str, list[str]]] = []
    current_name = ""
    current_body: list[str] = []

    for line in markdown_text.splitlines():
        match = re.match(r"^##\s+(.+)$", line.strip())
        heading = match.group(1).strip() if match else ""
        if heading in clause_names:
            if current_name:
                blocks.append((current_name, current_body))
            current_name = heading
            current_body = []
            continue
        if current_name:
            current_body.append(line)

    if current_name:
        blocks.append((current_name, current_body))

    return [
        (name, "\n".join(body).strip())
        for name, body in blocks
        if "\n".join(body).strip()
    ]


def parse_markdown_sections(markdown_text: str) -> list[tuple[str, str]]:
    lines = markdown_text.strip().splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Overview"
    current_body: list[str] = []

    for line in lines:
        match = re.match(r"^#{1,3}\s+(.+)$", line.strip())
        if match:
            if current_body:
                sections.append((current_heading, current_body))
            current_heading = match.group(1).strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_heading, current_body))

    return [(heading, "\n".join(body).strip()) for heading, body in sections if "\n".join(body).strip()]


def render_section_card(heading: str, body: str) -> str:
    card_class = card_class_for_heading(heading)
    icon = icon_for_heading(heading)
    full = " full" if card_class in {"danger", "citation", "source"} or len(body) > 900 else ""
    return (
        f'<section class="brief-card {card_class}{full}">'
        f"<h3><span class=\"icon-dot\">{html.escape(icon)}</span>{html.escape(heading)}</h3>"
        f"{source_body_to_html(body) if card_class == 'source' else markdown_body_to_html(body)}"
        "</section>"
    )


def card_class_for_heading(heading: str) -> str:
    lowered = heading.lower()
    if any(term in lowered for term in ["risk", "watch", "warning", "important"]):
        return "warning"
    if any(term in lowered for term in ["liability", "termination", "non-refundable", "penalty"]):
        return "danger"
    if "citation" in lowered or "supporting" in lowered:
        return "citation"
    if "source text" in lowered:
        return "source"
    if "clarify" in lowered or "question" in lowered:
        return "clarify"
    return "standard"


def icon_for_heading(heading: str) -> str:
    lowered = heading.lower()
    if any(term in lowered for term in ["risk", "watch", "warning", "important"]):
        return "!"
    if "citation" in lowered or "supporting" in lowered:
        return "§"
    if "source text" in lowered:
        return "↗"
    if "clarify" in lowered or "question" in lowered:
        return "?"
    if "summary" in lowered or "answer" in lowered:
        return "✓"
    if "obligation" in lowered or "term" in lowered:
        return "•"
    return "i"


def markdown_body_to_html(body: str) -> str:
    lines = [line.rstrip() for line in body.splitlines()]
    html_parts = []
    list_items = []

    def flush_list():
        if list_items:
            html_parts.append("<ul>" + "".join(f"<li{attrs}>{item}</li>" for attrs, item in list_items) + "</ul>")
            list_items.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_list()
            continue
        bullet = re.match(r"^[-*]\s+(.+)$", stripped)
        if bullet:
            item_text = bullet.group(1)
            chunk_match = re.match(r"\[Chunk (\d+)\]\(#chunk-\d+\)", item_text)
            attrs = f' id="chunk-{chunk_match.group(1)}"' if chunk_match else ""
            list_items.append((attrs, format_inline(item_text)))
            continue
        flush_list()
        html_parts.append(f"<p>{format_inline(stripped)}</p>")

    flush_list()
    return "".join(html_parts)


def source_body_to_html(body: str) -> str:
    chips = []
    for line in body.splitlines():
        stripped = line.strip()
        match = re.match(
            r"^[-*]\s+\[Chunk (\d+)\]\(#chunk-\d+\)\s+\|\s+([^:]+):\s+(.+)$",
            stripped,
        )
        if not match:
            continue
        chunk_num, meta, excerpt = match.groups()
        chips.append(
            f"""
            <div class="source-chip" id="chunk-{html.escape(chunk_num)}">
              <div class="source-chip-header">
                <a class="source-chip-label" href="#chunk-{html.escape(chunk_num)}">Chunk {html.escape(chunk_num)}</a>
                <span class="source-chip-meta">{html.escape(meta)}</span>
              </div>
              <p>{format_inline(excerpt)}</p>
            </div>
            """
        )
    if not chips:
        return markdown_body_to_html(body)
    return f'<div class="source-list">{"".join(chips)}</div>'


def format_inline(text: str) -> str:
    safe = html.escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(
        r"\[(Chunk \d+)\]\(#chunk-(\d+)\)",
        r'<a href="#chunk-\2">\1</a>',
        safe,
    )
    safe = re.sub(r"\[(Chunk (\d+))\]", r'<a href="#chunk-\2">\1</a>', safe)
    return safe


def get_server_port() -> int:
    configured = os.getenv("GRADIO_SERVER_PORT") or os.getenv("PORT")
    if configured:
        return int(configured)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


with gr.Blocks(title=APP_TITLE) as demo:
    doc_state = gr.State()

    gr.HTML(
        """
        <div class="app-shell">
          <div class="hero">
            <div>
              <h1>Contract Clarity</h1>
              <p>Upload a contract and get a flyer-style brief with plain-English explanations, warning callouts, key clauses, and source-backed answers.</p>
            </div>
            <div class="hero-badge">Consumer Contract Review</div>
          </div>
        </div>
        """
    )

    with gr.Row(elem_classes=["app-shell", "control-strip"]):
        with gr.Column(scale=4):
            file_input = gr.File(
                label="Contract",
                file_types=[".pdf", ".docx", ".html", ".htm"],
            )
        with gr.Column(scale=1, min_width=170):
            upload_button = gr.Button("Process", variant="primary")

    with gr.Row(elem_classes=["app-shell"]):
        upload_status = gr.Textbox(
            label="Status",
            lines=4,
            interactive=False,
            elem_classes=["status-box"],
        )

    with gr.Row(elem_classes=["app-shell"]):
        with gr.Column(scale=8):
            with gr.Row():
                summary_button = gr.Button("Generate Flyer Brief", variant="primary")
                clauses_button = gr.Button("Clause Watchlist")
            brief_output = gr.HTML(
                render_empty_brief(
                    "Flyer Brief",
                    "Upload a contract, then generate a polished one-page review.",
                ),
                elem_classes=["flyer-output"],
            )
        with gr.Column(scale=3):
            gr.HTML(
                """
                <div class="qa-panel">
                  <h3 class="side-title">Ask The Contract</h3>
                  <p style="color: var(--muted); margin-top: 0;">Ask follow-up questions and get answers grounded in the uploaded document.</p>
                </div>
                """
            )
            question_box = gr.Textbox(
                label="Question",
                placeholder="What should I watch out for before signing?",
                lines=4,
            )
            ask_button = gr.Button("Ask Question", variant="primary")
            answer_output = gr.HTML(
                render_empty_brief("Question Box", "Your answer will appear here as a mini brief."),
            )
            preview_output = gr.HTML(preview_html())
            gr.HTML(
                """
                <p class="disclaimer">
                  This tool explains contract text in plain English. It does not provide legal advice or replace an attorney.
                </p>
                """
            )

    upload_button.click(
        fn=upload_contract,
        inputs=[file_input],
        outputs=[upload_status, doc_state, preview_output],
    )
    summary_button.click(fn=summarize_contract, inputs=[doc_state], outputs=[brief_output])
    clauses_button.click(fn=extract_key_clauses, inputs=[doc_state], outputs=[brief_output])
    ask_button.click(fn=ask_question, inputs=[question_box, doc_state], outputs=[answer_output])


if __name__ == "__main__":
    default_server = "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1"
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", default_server),
        server_port=get_server_port(),
        css=CSS,
        theme=gr.themes.Soft(),
    )
