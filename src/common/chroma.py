from langchain_community.vectorstores.chroma import Chroma

def get_chroma_vector_store(collection_name, embeddings, vectordb_folder):
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=vectordb_folder,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("Chroma vector store created")
    vector_store_data = vector_store.get()
    print("Length of vector store: ", len(vector_store_data['ids']))
    return vector_store