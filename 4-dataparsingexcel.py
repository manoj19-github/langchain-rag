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
)
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from docx import Document as DocxDocument

from students import students


os.makedirs("data/csv", exist_ok=True)


df = pd.DataFrame(students)
df.to_csv("data/csv/students.csv", index=False)


with pd.ExcelWriter("data/csv/students.xlsx") as writer:
    df.to_excel(writer, sheet_name="Sheet1",index=False)
    summary_data = {
        "Metric": [
            "Total Students",
            "School",
            "Average Age",
            "Average Semester",
            "Average CGPA",
            "Average Attendance",
        ],
        "Value": [
            len(students),
            "Indian Institute of Technology, Bombay",
            round(df["age"].mean(), 2),
            round(df["semester"].mean(), 2),
            round(df["cgpa"].mean(), 2),
            round(df["attendance"].mean(), 2),
        ],
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary",index=False)
    

csv_loader = CSVLoader(
    file_path="data/csv/students.csv",
    encoding="utf-8",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
    }
)
csv_docs = csv_loader.load()
print(f"Loaded {len(csv_docs)} documents from the CSV file.")
print("metadata ",csv_docs[0].metadata)
print("page_content ",csv_docs[0].page_content)

# excel_loader = ExcelLoader(
#     file_path="data/csv/students.xlsx",
#     sheet_name="Sheet1",
# )


