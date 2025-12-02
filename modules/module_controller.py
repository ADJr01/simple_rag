from modules.Parsers.text_parser import TextParser
from langchain_community.vectorstores import chroma




storage_dir = '/home/ad/Desktop/Project/simple_rag/modules/storage/'
chroma_persist = '/home/ad/Desktop/Project/simple_rag/modules/db/'
collection_name = 'rag_memory'

def load_storage():
    text_parser = TextParser()
    text_parser.load_from(storage_dir,True)
    return text_parser.process()

def main():
    load_storage()

