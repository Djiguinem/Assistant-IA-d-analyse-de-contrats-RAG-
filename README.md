# ⚖️ Assistant d’Analyse de Contrats Juridiques (RAG)

Système basé sur l’intelligence artificielle permettant d’analyser des contrats juridiques, d’extraire automatiquement des clauses importantes et de répondre à des questions en langage naturel grâce à une architecture **RAG (Retrieval-Augmented Generation)**.

---

## Objectif

Ce projet vise à automatiser l’analyse de contrats en combinant :
- le **Traitement Automatique du Langage Naturel (NLP)**
- les **modèles de langage (LLM)**
- la **recherche sémantique**

L’utilisateur peut :
1. Importer un contrat PDF  
2. Extraire automatiquement les clauses clés  
3. Poser des questions sur le document  
4. Obtenir des réponses contextualisées  

---

## Architecture du système

Le pipeline se décompose en 3 grandes étapes :

---

### 1️⃣ Extraction des clauses

- **Entrée :** Contrat PDF  
- **Outil :** PyPDF2  

**Étapes :**
- Extraction du texte brut depuis le PDF  
- Analyse via un modèle de langage (**Gemma3 via Ollama**)  
- Identification des clauses importantes (responsabilité, résiliation, audit, etc.)

**Sortie :** Fichier JSON structuré  

Exemple :
json
{
  "responsabilite": "...",
  "resiliation": "...",
  "audit": "..."
}

### 2️⃣ RAG – Recherche augmentée
- Indexation (offline)
  Découpage du texte (chunking)
  Transformation en vecteurs (embeddings)
  Stockage dans une base vectorielle (ChromaDB)
🔎 Recherche (online)
  Transformation de la question en embedding
  Calcul de similarité (cosine similarity)
  Récupération des passages les plus pertinents (Top-k)
### 3️⃣ Génération de réponse
Le modèle de langage reçoit :
-les clauses extraites
-le contexte récupéré (RAG)
-la question utilisateur
---> Il génère une réponse argumentée, contextualisée et compréhensible.
### 🖥️ Interface utilisateur
Interface développée avec Gradio permettant :
l’upload de contrats PDF
la visualisation des clauses extraites
une interaction sous forme de chat

### Technologies utilisées:
- Python
- LangChain (orchestration du pipeline RAG)
- Ollama + Gemma3 (LLM local)
- ChromaDB (base vectorielle)
- PyPDF2 (lecture PDF)
- Gradio (interface utilisateur)
## ⚙️ Installation rapide

```bash
git clone https://github.com/...
cd project

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

ollama pull gemma3
