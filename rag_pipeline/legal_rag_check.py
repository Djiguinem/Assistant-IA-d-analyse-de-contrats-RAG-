import json

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# -----------------------------
# LLM
# -----------------------------

llm = ChatOllama(
    model="gemma3:latest",
    temperature=0
)

# -----------------------------
# RAG configuration
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="rag_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# Chargement des clauses
# -----------------------------

with open("extracted_clauses.json") as f:
    clauses = json.load(f)

# -----------------------------
# Analyse juridique
# -----------------------------

results = []

for clause_type, clause_text in clauses.items():

    if clause_text is None:
        continue

    print("\nAnalyse :", clause_type)

    docs = retriever.invoke(clause_text)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a legal AI compliance assistant.

Contract clause:

{clause_text}

Relevant regulations:

{context}

Tasks:

1 Identify legal risks
2 Determine compliance level

Return ONLY JSON:

{{
 "risk_level": "LOW | MEDIUM | HIGH | CRITICAL",
 "law": "...",
 "issue": "...",
 "recommendation": "..."
}}
"""

    response = llm.invoke(prompt)

    try:

        text = response.content.strip()

        if text.startswith("```"):
            text = text.split("```")[1]

        text = text.replace("json", "", 1).strip()

        result = json.loads(text)

        result["clause_type"] = clause_type
        result["clause"] = clause_text

        results.append(result)

    except:

        print("Parsing error")
        print(response.content)

# -----------------------------
# Rapport final
# -----------------------------

print("\n===== LEGAL RISK REPORT =====\n")

for r in results:

    print("Clause type :", r["clause_type"])
    print("Risk level :", r["risk_level"])
    print("Law :", r["law"])
    print("Issue :", r["issue"])
    print("Recommendation :", r["recommendation"])
    print("\n------------------------\n")

# sauvegarde

with open("legal_risk_report.json", "w") as f:
    json.dump(results, f, indent=4)