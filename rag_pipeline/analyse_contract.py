from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
import json

def clean_json_response(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]

    text = text.replace("json", "", 1).strip()

    return text
# ----------------------------------------------------
# 1 Chargement du contrat pour un test sur le terminal
# ----------------------------------------------------

file_path = "data/contrats/master_service_agreement.pdf"

loader = PyPDFLoader(file_path)
documents = loader.load()

# -----------------------------
# 2 Découpage du contrat
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Nombre de sections analysées :", len(chunks))

# -----------------------------
# 3 LLM gemma3
# -----------------------------

llm = ChatOllama(
    model="gemma3:latest",   
    temperature=0
)

# -----------------------------
# 4 Prompt d'analyse
# -----------------------------

prompt_template = """
You are a legal AI assistant specialized in contract analysis.

From the following contract excerpt, extract information related to these clauses:

- Data hosting or data localisation
- Liability or responsibilities
- Subcontracting
- Audit rights
- Legal compliance or legal restrictions

Return the result strictly in JSON format:

{{
"data_hosting": "...",
"liability": "...",
"subcontracting": "...",
"audit_rights": "...",
"legal_compliance": "..."
}}

If a clause is not present, return null.

Contract excerpt:
{chunk}
"""

# -----------------------------
# 5 Analyse des chunks
# -----------------------------

all_results = []

for chunk in chunks:
    
    prompt = prompt_template.format(chunk=chunk.page_content)

    response = llm.invoke(prompt)

    try:
        clean_text = clean_json_response(response.content)

        json_result = json.loads(clean_text)

        # corriger les "null" strings
        for key, value in json_result.items():
            if value == "null":
                json_result[key] = None

        all_results.append(json_result)

    except Exception as e:
        print("⚠️ JSON parsing failed")
        print(response.content)
# -----------------------------
# 6 Fusion des résultats
# -----------------------------

final_result = {
    "data_hosting": None,
    "liability": None,
    "subcontracting": None,
    "audit_rights": None,
    "legal_compliance": None
}

for result in all_results:

    for key in final_result:

        if result.get(key) and not final_result[key]:
            final_result[key] = result[key]

# -----------------------------
# 7 Résultat final
# -----------------------------

print("\n===== CLAUSES EXTRAITES =====\n")
with open("extracted_clauses.json", "w") as f:
    json.dump(final_result, f, indent=4)
print(json.dumps(final_result, indent=4))