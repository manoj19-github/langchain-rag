import os 
import redis

from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import Annotated,List
from typing_extensions import TypedDict
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint,HuggingFaceEmbeddings
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
from langchain_core.output_parsers import StrOutputParser
import chromadb
from docx import Document as DocxDocument
from langchain_chroma import Chroma
import numpy as np
from typing import List

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough
load_dotenv()


# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#       task="conversational",
#     huggingfacehub_api_token=os.getenv("HF_TOKEN"),
# )



llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)

llm = ChatHuggingFace(llm=llm_endpoint)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


dir_loader = DirectoryLoader("data/test_files",glob="**/*.txt",loader_cls=TextLoader,loader_kwargs={"encoding":"utf-8"})

documents = dir_loader.load()

# Method 1 : Character Text Splitter
# text = documents[0].page_content

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n","."," ",""],
    length_function=len,
)

chunks = text_splitter.split_documents(documents)




print(" ---------------------------------------------------",end="\n\n")


# print("chunks: ",chunks)


# chunk_texts = [doc.page_content for doc in chunks]
# # Embedding models
# chunk_embeddings = embeddings.embed_documents(
#     chunk_texts
# )


# Chromadb store

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="rag_collection",
)

# similar_docs = vector_store.similarity_search(" Types of Artificial Intelligence",k=3)
# #similar_docs = vector_store.similarity_search_with_score(" Types of Artificial Intelligence",k=3)


# for i , doc in enumerate(similar_docs):
#     print(f"\nResult {i}")
#     print(doc)
# print("similar_docs: ",similar_docs)
# print("vector_store: ",vector_store._collection.count())


retriever = vector_store.as_retriever(search_kwargs={"k":3})

systemPrompt = """
You are a assitant for question answering tasks. 
If you don't know the answer, just say you don't 
know use three sentences maximum and keep answer concise.

Context: {context}
"""
question = input("Enter your question: ")


prompt = ChatPromptTemplate.from_messages([
    ("system", systemPrompt),
    ("human", "{question}"),
    
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)






# rag_chain = create_retrieval_chain(
#     retriever=retriever,
#     combine_documents_chain=document_chain)



results = vector_store.similarity_search_with_score(question, k=3)

print("\nRetrieved Documents\n")

# for i, (doc, score) in enumerate(results, start=1):
#     print(f"\nResult {i}")
#     print(f"Similarity Score: {score}")
#     print(f"Metadata: {doc.metadata}")
#     print(doc.page_content)






rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)




print("---------------------------------------------------------------------------------------------",end="\n\n")
result = rag_chain.invoke(question)
print(result)
