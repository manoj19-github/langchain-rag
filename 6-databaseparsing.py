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
from langchain_community.utilities import SQLDatabase
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_core.documents import Document
import sqlite3


conn  = sqlite3.connect("data/databases/company.db")
cursor = conn.cursor()

cursor.execute(""" create table if not exists employees 
               (id INTEGER PRIMARY KEY, name TEXT,department TEXT,salary REAL)  
""")

cursor.execute(""" create table if not exists projects 
               (id INTEGER PRIMARY KEY, name TEXT,status TEXT,lead_id INTEGER)
""")




employees=[
    {"id":1,"name":"John","department":"HR","salary":10000},
    {"id":2,"name":"Mary","department":"HR","salary":10000},
    {"id":3,"name":"David","department":"IT","salary":10000},
    {"id":4,"name":"Peter","department":"IT","salary":10000},
    {"id":5,"name":"Sarah","department":"IT","salary":10000},
    {"id":6,"name":"Mike","department":"IT","salary":10000},
    {"id":7,"name":"Jane","department":"IT","salary":10000},
    {"id":8,"name":"Bob","department":"IT","salary":10000},
    {"id":9,"name":"Alice","department":"IT","salary":10000},
    {"id":10,"name":"Carol","department":"IT","salary":10000},
    {"id":11,"name":"Eve","department":"IT","salary":10000},
    {"id":12,"name":"Frank","department":"IT","salary":10000},
]

projects=[
    {"id":1,"name":"Project 1","status":"In Progress","lead_id":1},
    {"id":2,"name":"Project 2","status":"In Progress","lead_id":2},
    {"id":3,"name":"Project 3","status":"In Progress","lead_id":3},
    {"id":4,"name":"Project 4","status":"In Progress","lead_id":4},
    {"id":5,"name":"Project 5","status":"In Progress","lead_id":5},
    {"id":6,"name":"Project 6","status":"In Progress","lead_id":6},
    {"id":7,"name":"Project 7","status":"In Progress","lead_id":7},
    {"id":8,"name":"Project 8","status":"In Progress","lead_id":8},
    {"id":9,"name":"Project 9","status":"In Progress","lead_id":9},        
]


cursor.executemany(
    """
    INSERT OR REPLACE INTO employees (id, name, department, salary)
    VALUES (:id, :name, :department, :salary)
    """,
    employees
)
cursor.executemany(
    """
    INSERT OR REPLACE INTO projects (id, name, status, lead_id)
    VALUES (:id, :name, :status, :lead_id)
    """,
    projects
)
conn.commit()       

conn.close()


db = SQLDatabase.from_uri("sqlite:///data/databases/company.db")


print("------------------------------ ")

print(f"tables in the database: {db.get_usable_table_names()}")
print("----------------------------------------------------- ")
print(db.get_table_info())



def sql_to_documents(db_path: str) -> List[Document]:
    """ 
    converts a sql database to a list of documents
    """
    conn  = sqlite3.connect(db_path)
    cursor = conn.cursor()
    documents = []
    cursor.execute("""select name from sqlite_master where type='table'""")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        
        cursor.execute(f"select * from {table_name}")
        rows = cursor.fetchall()
        table_content=f"Table: {table_name}\n"
        table_content+=f"Columns : {', '.join(column_names)}\n"
        table_content+=f"Rows: {len(rows)}\n"
        # Sample Records 
        table_content+="Sample Records:\n"
        for row in rows:
            record = dict(zip(column_names, row))
            table_content+=f"{record}\n"
        doc = Document(
            page_content=table_content, 
            metadata={
                           "source": db_path,
                           "table_name": table_name,
                           "column_names": column_names,
                           "row_count": len(rows),
                           "data_type": type(row)
                        }
                    )
        documents.append(doc)
    return documents
    



documents = sql_to_documents("data/databases/company.db")
print(documents)
    




# class Employee(TypedDict):
#     name: str
#     department: str
#     salary: float

# class Project(TypedDict):
#     name: str
#     status: str
#     lead_id: int

# def get_employees():
#     conn  = sqlite3.connect("data/databases/company.db")
#     cursor = conn.cursor()
#     cursor.execute("select * from employees")
#     employees = cursor.fetchall()
#     return employees