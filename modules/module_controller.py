from modules.Parsers.text_parser import TextParser
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings



storage_dir = '/home/ad/Desktop/Project/simple_rag/modules/storage/'
chroma_persist = '/home/ad/Desktop/Project/simple_rag/modules/db/'
collection_name = 'rag_memory'

def load_storage():
    text_parser = TextParser()
    text_parser.load_from(storage_dir,True)
    return text_parser.process()

def get_embeddings():
    return OllamaEmbeddings(model='qwen3-embedding:0.6b')

def get_vector_store(documents,embeddings,collection_name,persist_dir):
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_dir=persist_dir,
    )



def main():
    docs = load_storage()
    vector_store = get_vector_store(documents=docs,embeddings=get_embeddings(),collection_name=collection_name,persist_dir=chroma_persist)
    # setting up RAG
    retrival = vector_store.as_retriever(
        search_kwargs={"k":2},
    )


