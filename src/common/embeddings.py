from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
load_dotenv()

def getEmbeddings(embeddings_provider, embeddings_model):
    if embeddings_provider == "huggingface":
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        print(f"[Info] ~ Loaded huggingface embeddings: {embeddings_model}")
    elif embeddings_provider == "local_embeddings":
        model_name = embeddings_model
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print(f"[Info] ~ Loaded local embeddings: {embeddings_model}")
    return embeddings
