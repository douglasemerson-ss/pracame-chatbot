from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

#from langchain.text_splitter import RecursiveCharacterTextSplitter

PASTA_BASE = "base"

def criar_db():
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos) #dividir os documentos em pedaços de texto(chunks)
    vetorizar_chunks(chunks) #vetorizar os chunks com o processo de embedding

def carregar_documentos():      
    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf")        
    documentos = carregador.load()
    return documentos

def dividir_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = separador_documentos.split_documents(documentos)
    print(len(chunks))
    return chunks

def vetorizar_chunks(chunks):
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="db_2")
    print("Banco de dados criado")
        
criar_db()
