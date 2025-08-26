import gradio as gr
from setfit import SetFitModel
from pathlib import Path
import os

token = os.environ["HF_TOKEN"]

models = {
    'target': SetFitModel.from_pretrained('munterk/trvinfra_target', token=token),
    'nature': SetFitModel.from_pretrained('munterk/trvinfra_nature', token=token),
    'interpretability': SetFitModel.from_pretrained('munterk/trvinfra_interpretability', token=token),
    'reference': SetFitModel.from_pretrained('munterk/trvinfra_reference', token=token)
}


def classify(text):
    if not text.strip():
        return ["No input"] * 4

    result = []

    for name, model in models.items():
        probs = model.predict_proba([text])[0]
        print(f'{name}: {probs}')
        id2label = model.id2label
        result.append({id2label[i]: float(p) for i, p in enumerate(probs)})

    return result


ARTICLE_MD = """
# Classification Scheme Summary

## 1. Target
The **target** of a requirement identifies the general field to which it applies, influencing how compliance is verified.

* **Product Requirement**: Refers to an artifact delivered by the supplier (digital or physical).
* **Process Requirement**: Specifies how a product should be produced, including constraints and procedures.
* **Documentation Requirement**: Defines what information must be recorded and delivered.

## 2. Nature
* **Qualitative**: Non-numeric, descriptive information.
* **Quantitative**: Numerical values or measurable data.
* **Mixture**: Both qualitative and quantitative elements.

## 3. Interpretability
* **Natural Ambiguity**: Intentional flexibility.
* **Artificial Ambiguity**: Poor formulation (language, vagueness, incompleteness, lexical issues).

## 4. Reference
* **Local Reference**: Within same document.
* **Internal Reference**: Within same organization.
* **External Reference**: External standards/laws.
"""

EXAMPLES = [
    ["Lager ska placeras med sitt centrum minst 200 mm och med bultcentrum minst 100 mm från underliggande konstruktions sida."],
    ["Utfartssignal ska vara placerad på driftplats, sista signal vid eller innanför driftplatsgräns."],
    ["Vid apparatkontroll av reläställverk, ska följande särskilt kontrolleras enligt ritning 1."],
    ["Drag i tillsatsrör ska vara större än 70 N."]
]

custom_css = """
#logo-container {
    text-align: center;
    margin-bottom: 20px;
}
#logo-container img {
    height: 80px;
}
"""

if __name__ == "__main__":
    gr.set_static_paths(paths=[Path.cwd().absolute()/"assets"])

    with gr.Blocks(title="Requirements Verifiability Classifier", css=custom_css) as demo:

        with gr.Row(elem_id="logo-container"):
            gr.HTML("<img src='/gradio_api/file=assets/bth.png' alt='BTH logo'>")

        gr.Markdown("# Requirements Verifiability Classifier")

        with gr.Row():
            with gr.Column(scale=3):
                req_inp = gr.Textbox(
                    label="Requirement",
                    lines=8,
                    autofocus=True,
                    placeholder="Paste a requirement here…"
                )
                submit = gr.Button("Classify", variant="primary")

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[req_inp],
                    label="Examples"
                )

            with gr.Column(scale=2):
                target = gr.Label(label="Target")
                nature = gr.Label(label="Nature")
                interp = gr.Label(label="Interpretability")
                reference = gr.Label(label="Reference")

        submit.click(
            fn=classify,
            inputs=[req_inp],
            outputs=[target, nature, interp, reference]
        )

        gr.Markdown(ARTICLE_MD)

        demo.launch(server_name="0.0.0.0", server_port=7860)
