from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


data_path = Path("data/regulations")

documents = []

# charger tous les PDF
for pdf in data_path.glob("*.pdf"):
    print(f"Loading {pdf}")
    loader = PyPDFLoader(str(pdf))
    documents.extend(loader.load())

print(f"{len(documents)} pages chargées")

# découpage du texte
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"{len(chunks)} chunks créés")

# embeddings gratuits
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# base vectorielle
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="rag_db"
)

vectorstore.persist()

print("Vector DB créée dans rag_db")