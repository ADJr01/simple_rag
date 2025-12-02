from modules.Parsers.text_parser import TextParser
from langchain_community.vectorstores import chroma
storage_dir = '/home/ad/Desktop/Project/simple_rag/modules/storage/'
collection_name = 'rag_memory'

def main():
    text_parser = TextParser()
    text_parser.load_from(storage_dir,True)
    t = text_parser.process()
    print(t)
