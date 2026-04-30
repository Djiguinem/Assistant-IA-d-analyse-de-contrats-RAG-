import gradio as gr
import json
import subprocess
from pathlib import Path
from PyPDF2 import PdfReader
from langchain_ollama import OllamaLLM

import gradio_client.utils as client_utils

original_get_type = client_utils.get_type

def patched_get_type(schema):
    if isinstance(schema, bool):
        return "boolean"
    return original_get_type(schema)

client_utils.get_type = patched_get_type

# --------------------------
# Configuration
# --------------------------
clauses_file = Path("extracted_clauses.json")

# LLM Ollama
llm = OllamaLLM(model="gemma3:latest")


# --------------------------
# Fonction principale
# --------------------------
def process_contract(file, question, chat_history):

    if file is None:
        return "", "Please upload a PDF.", chat_history

    # --- CORRECTION CHEMIN PDF (EmptyFileError) ---
    file_path = file.name if hasattr(file, 'name') else file

    # --------------------------
    # 1. Lire le PDF
    # --------------------------
    pdf = PdfReader(file_path)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    Path("current_contract.txt").write_text(text)

    # --------------------------
    # 2. Extraction des clauses
    # --------------------------
    try:
        subprocess.run(
            ["python", "rag_pipeline/analyse_contract.py"],
            check=True
        )

        with open("extracted_clauses.json") as f:
            clauses = json.load(f)

        clauses_display = json.dumps(clauses, indent=2)

    except Exception as e:
        return "", f"Clause extraction failed: {e}", chat_history

    # --------------------------
    # 3. Contexte
    # --------------------------
    context_text = "\n".join(
        [f"{k}: {v}" for k, v in clauses.items() if v != "null"]
    )

    # --------------------------
    # 4. Question utilisateur
    # --------------------------
    if question:
        full_prompt = f"""
        Context clauses:
        {context_text}

        Question: {question}
        Answer:
        """

        try:
            response = llm.invoke(full_prompt)
        except Exception as e:
            response = f"LLM error: {e}"

        chat_history = chat_history + [[question, response]]

    return "", clauses_display, chat_history


# --------------------------
# Interface Gradio
# --------------------------
with gr.Blocks() as demo:

    gr.Markdown("# ⚖️ Legal Contract Assistant")
    gr.Markdown("Upload a contract PDF and ask questions about extracted clauses.")

    with gr.Row():
        file_input = gr.File(label="Upload Contract PDF")
        question_input = gr.Textbox(label="Your question")

    analyze_btn = gr.Button("Process & Ask")

    gr.Markdown("## 📊 Extracted Clauses")
    clauses_output = gr.Code(language="json")

    gr.Markdown("## 💬 Chat")
    
    
    chatbot = gr.Chatbot()

    state = gr.State([])

    analyze_btn.click(
        fn=process_contract,
        inputs=[file_input, question_input, state],
        outputs=[question_input, clauses_output, chatbot]
    )

demo.launch(share=True)