from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# chargement des embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# chargement de la base vectorielle
vectorstore = Chroma(
    persist_directory="rag_db",
    embedding_function=embeddings
)

# creation du retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

while True:
    query = input("\nPose ta question (ou 'exit'): ")

    if query.lower() == "exit":
        break

    docs = retriever.get_relevant_documents(query)

    print("\nDocuments trouvés :\n")

    for i, doc in enumerate(docs):
        print(f"Result {i+1}")
        print(doc.page_content[:500])
        print("------")