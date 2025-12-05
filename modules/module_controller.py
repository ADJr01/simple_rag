from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from modules.Parsers.text_parser import TextParser
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings



storage_dir = 'D:\\Projects\\Personal\\LLM\\simple_rag\\modules\\storage'
chroma_persist = 'D:\\Projects\\Personal\\LLM\\simple_rag\\modules\\db'
collection_name = 'rag_memory'
sys_prompt = '''
    You're an helpful assistant for question answering task.
    use the following pieces of retrieved context to answer the question:

    if you don't know the answer then simply reply You Can't help with that question.

    if users query is complex then Break the
    query and divide them into logical sub-queries 
    then solve the problem step by step.

    Context: {context}
'''

prompt = ChatPromptTemplate.from_messages([
    {"role": "system", "content": sys_prompt},
    {"role": "human", "content": "{input}"}
])

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
        persist_directory=persist_dir,
    )


def create_llm():
    return ChatOllama(
        model='deepseek-r1:8b',
        reasoning=True
    )



def main():
    docs = load_storage()
    vector_store = get_vector_store(documents=docs,embeddings=get_embeddings(),collection_name=collection_name,persist_dir=chroma_persist)
    # setting up RAG
    retrival = vector_store.as_retriever(
        search_kwargs={"k":2},
    )
    llm = create_llm()
    document_chain = create_stuff_documents_chain(llm=llm,prompt=prompt)
    rag_chain = create_retrieval_chain(retrival,document_chain)
    str=str=input("Enter your question: ")
    while str.lower().strip() != 'q' or str.lower().strip() != 'exit':
        str = str.strip()
        response = rag_chain.invoke({"input":str})
        print(response["answer"])
        str = input("Enter your question: ")


