import os 



from dotenv import load_dotenv
from typing import Annotated,List
from typing_extensions import TypedDict
from langchain_huggingface import HuggingFaceEndpoint
import re
import unicodedata
# from langchain_tavily  import TavilySearch
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,DirectoryLoader,PyPDFLoader,PyMuPDFLoader,UnstructuredPDFLoader,
    OnlinePDFLoader,PDFMinerLoader,PDFPlumberLoader,
    UnstructuredWordDocumentLoader,Docx2txtLoader
)
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from docx import Document as DocxDocument


# Method 1: Using the Docx2txtLoader
def load_docx_with_docx2txtloader(file_path):
    try:
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from the DOCX file.")
        print("metadata ",documents[0].metadata)
        print("page_content ",documents[0].page_content)
        return documents
    except Exception as e:
        print(f"Error loading DOCX file: {e}")
        return None
    
# Method 2: Using the UnstructuredWordDocumentLoader



# word_docs = load_docx_with_docx2txtloader("data/word_files/proposal.docx")

def load_docx_with_unstructuredworddocumentloader(file_path):
    try:
        loader = UnstructuredWordDocumentLoader(file_path,mode="elements")
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from the DOCX file.")
        print("metadata ",documents[0].metadata)
        print("page_content ",documents[0].page_content)
        return documents
    except Exception as e:
        print(f"Error loading DOCX file: {e}")
        return None

word_docs = load_docx_with_unstructuredworddocumentloader("data/word_files/proposal.docx")