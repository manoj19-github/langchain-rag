import os
import pandas as pd


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
    UnstructuredWordDocumentLoader,Docx2txtLoader,
    CSVLoader,
    JSONLoader
)
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from docx import Document as DocxDocument

from students import students

json_path = os.path.abspath("data/json_files/employees.json")
print(json_path)

employee_loader = JSONLoader(
    file_path=json_path,
    jq_schema=".employees[]",
    text_content=False
)

employee_docs = employee_loader.load()
print(f"Loaded {len(employee_docs)} documents from the JSON file.")
print(f"First employee: {employee_docs[0].page_content[:200]}")

