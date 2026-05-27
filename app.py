import gradio as gr
from transformers import pipeline

# Load the NER pipeline once at startup
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple"   # merges subword tokens automatically
)

# Label color mapping for UI
LABEL_COLORS = {
    "PER":  "#FFD700",   # Person       → Gold
    "ORG":  "#87CEEB",   # Organization → Sky Blue
    "LOC":  "#98FB98",   # Location     → Pale Green
    "DATE": "#FFB6C1",   # Date         → Light Pink  (if model emits it)
    "MISC": "#DDA0DD",   # Miscellaneous→ Plum
}

def format_entities_table(entities: list) -> str:
    """Build an HTML table of detected entities."""
    if not entities:
        return "<p style='color:gray'>No named entities detected.</p>"

    rows = ""
    for ent in entities:
        label = ent["entity_group"]
        color = LABEL_COLORS.get(label, "#E0E0E0")
        rows += f"""
        <tr>
            <td><span style='background:{color};padding:2px 8px;
                border-radius:4px;font-weight:bold'>{label}</span></td>
            <td>{ent['word']}</td>
            <td>{ent['score']:.2%}</td>
            <td>{ent['start']} – {ent['end']}</td>
        </tr>"""

    return f"""
    <table style='width:100%;border-collapse:collapse;font-family:monospace'>
      <thead>
        <tr style='background:#333;color:white;text-align:left'>
          <th style='padding:8px'>Entity Type</th>
          <th style='padding:8px'>Text</th>
          <th style='padding:8px'>Confidence</th>
          <th style='padding:8px'>Position</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def highlight_text(text: str, entities: list) -> str:
    """Return the original text with entity spans highlighted."""
    if not entities:
        return text

    result = ""
    prev = 0
    for ent in sorted(entities, key=lambda x: x["start"]):
        s, e = ent["start"], ent["end"]
        label = ent["entity_group"]
        color = LABEL_COLORS.get(label, "#E0E0E0")
        result += text[prev:s]                          # plain text before entity
        result += (
            f"<mark style='background:{color};padding:0 3px;"
            f"border-radius:3px' title='{label}'>"
            f"{text[s:e]}</mark>"
        )
        prev = e
    result += text[prev:]                               # remaining plain text
    return f"<div style='line-height:2;font-size:16px'>{result}</div>"


def run_ner(text: str):
    """Main function called by Gradio."""
    if not text.strip():
        return "<p style='color:red'>Please enter some text.</p>", ""

    entities = ner_pipeline(text)

    highlighted = highlight_text(text, entities)
    table      = format_entities_table(entities)

    return highlighted, table


# ── Gradio UI ────────────────────────────────────────────────────────────────
sample_texts = [
    "Barack Obama was born in Hawaii and studied at Harvard University before becoming the 44th President of the United States.",
    "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California in 1976.",
    "The Eiffel Tower in Paris was designed by Gustave Eiffel and completed on March 31, 1889.",
]

with gr.Blocks(title="NER — Named Entity Recognizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 Named Entity Recognition (NER)
    Powered by **BERT** fine-tuned on CoNLL-2003 via Hugging Face Transformers.
    Detects: `PER` (Person) · `ORG` (Organization) · `LOC` (Location) · `MISC`
    """)

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Type or paste text here…",
                lines=6
            )
            run_btn   = gr.Button("Analyze", variant="primary")
            gr.Examples(examples=sample_texts, inputs=text_input)

        with gr.Column(scale=1):
            highlighted_output = gr.HTML(label="Highlighted Text")
            table_output       = gr.HTML(label="Entity Table")

    run_btn.click(fn=run_ner, inputs=text_input,
                  outputs=[highlighted_output, table_output])
    text_input.submit(fn=run_ner, inputs=text_input,
                      outputs=[highlighted_output, table_output])

demo.launch()