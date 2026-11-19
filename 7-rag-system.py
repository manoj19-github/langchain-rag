import os 
import redis

from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import Annotated,List
from typing_extensions import TypedDict
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
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



# 100% Free Hugging Face model running locally via pipeline (like your embedding model)
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 256,
        "temperature": 0.1,
        "do_sample": True,
        "return_full_text": False,
    },
)



embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


dir_loader = DirectoryLoader("data/test_files",glob="**/*.txt",loader_cls=TextLoader,loader_kwargs={"encoding":"utf-8"})

documents = dir_loader.load()

# Method 1 : Character Text Splitter
# text = documents[0].page_content


# --------------------------------------------------  Text parsing   -----------------------------------------------------------------------#

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


# --------------------------------------------  Chroma  Vector Store       ---------------------------------------------------


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


#  -------------------------------------------------   Basic Retriever     -----------------------------------------------------

retriever = vector_store.as_retriever(search_kwargs={"k": 3}, search_type="similarity")

def format_docs(docs):
    """Format documents for insertion into prompt"""
    return "\n\n".join(doc.page_content for doc in docs)


# ------------------------------------------ 1. History-Aware Retriever Sub-chain ------------------------------------------

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# Sub-chain: reformulates query using history -> passes standalone query to retriever -> formats docs
history_aware_retriever = (
    contextualize_q_prompt
    | llm
    | StrOutputParser()
    | retriever
    | format_docs
)


# ------------------------------------------ 2. Conversational QA RAG Chain ------------------------------------------

qa_system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say you don't know. 
Keep the answer concise (three sentences maximum).

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

rag_chain = (
    RunnablePassthrough.assign(context=history_aware_retriever)
    | qa_prompt
    | llm
    | StrOutputParser()
)


# ------------------------------------------ 3. Interactive Multi-Turn Chat Loop ------------------------------------------

chat_history = []

print("=" * 60)
print("Conversational RAG initialized! Type 'exit' or 'quit' to stop.")
print("=" * 60)

while True:
    question = input("\nEnter your question: ").strip()
    if not question or question.lower() in ["exit", "quit"]:
        print("Exiting chat. Goodbye!")
        break

    result = rag_chain.invoke({
        "question": question,
        "chat_history": chat_history,
    })

    print(f"\nAnswer: {result}")

    # Append current turn to chat history for subsequent queries
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=result))
