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
from langchain_community.document_loaders import (TextLoader,DirectoryLoader,PyPDFLoader,PyMuPDFLoader,UnstructuredPDFLoader,OnlinePDFLoader,PDFMinerLoader,PDFPlumberLoader)
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

# PyPDFLoader: A loader that uses the PyPDF2 library to load PDF files.

def load_pdf_with_pypdfloader(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading PDF file: {e}")
        return None

def load_pdf_with_pymupdfloader(file_path):
    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading PDF file: {e}")
        return None
    
def load_pdf_with_unstructuredpdfloader(file_path):
    try:
        loader = UnstructuredPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading PDF file: {e}")
        return None
    



class SmartPDFProcessor:
    """ Advanced PDF processing with error handling"""
    def __init__(self,chunk_size=1000,chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n","\n"," "],
        )


    def _clean_pdf_text(self,text: str) -> str:
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text)

        # Remove invisible characters
        text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

        # Fix ligatures
        text = (
            text.replace("ﬁ", "fi")
                .replace("ﬂ", "fl")
                .replace("ﬀ", "ff")
                .replace("ﬃ", "ffi")
                .replace("ﬄ", "ffl")
        )

        # Fix hyphenated line breaks
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Normalize line endings
        text = text.replace("\r", "\n")

        # Collapse multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Replace remaining newlines with spaces
        text = re.sub(r"\n", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove spaces before punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        return text.strip()


        
    def process_pdf(self,pdf_path:str)->List[Document]:
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            process_chunks=[]
            for page_num,page in enumerate(pages):
                cleaned_text = self._clean_pdf_text(page.page_content)
                if len(cleaned_text.strip()) <50:
                    continue
                
                chunks = self.text_splitter.create_documents(
                    texts=[cleaned_text],
                    metadatas=[{
                        **page.metadata,
                        "page_num":page_num + 1,
                        "total_pages":len(pages),
                        "chunk_method":"smart_pdf_processor",
                        "chunk_count":len(cleaned_text)
                    }]
                )
                
                process_chunks.extend(chunks)
            return process_chunks
                
            
        except Exception as e:
            print(f"Error loading PDF file: {e}")
            return None

# documents = load_pdf_with_pypdfloader("data/pdf/attention.pdf")
# documents = load_pdf_with_pymupdfloader("data/pdf/attention.pdf")
# print(f"Loaded {len(documents)} documents from the PDF file.")
# print("metadata ",documents[0].metadata)
# print("page_content ",documents[0].page_content)

preProcessor = SmartPDFProcessor()

#   process a pdf 
smart_pdf_processor = preProcessor.process_pdf("data/pdf/attention.pdf")
print(f"Processed {len(smart_pdf_processor)} chunks from the PDF file.")
print("metadata ",smart_pdf_processor[0].metadata)
print("page_content ",smart_pdf_processor[0].page_content)





