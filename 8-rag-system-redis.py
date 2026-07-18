import os 
import redis
import json
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
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
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import uuid


from langchain_core.runnables import RunnablePassthrough
load_dotenv()

chat_history = []
# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#       task="conversational",
#     huggingfacehub_api_token=os.getenv("HF_TOKEN"),
# )



llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0
)



# llm_endpoint = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     huggingfacehub_api_token=os.getenv("HF_TOKEN"),
# )

# llm = ChatHuggingFace(llm=llm_endpoint)

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


# prompt = ChatPromptTemplate.from_messages([
#     ("system", systemPrompt),
#     ("human", "{question}"),
    
# ])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)




redis_client = redis.Redis(
    host="localhost",
    port=6379,
    password="yourStrongPassword123",
    decode_responses=True,
)

if redis_client.ping():
    print("Redis is running")
    
else:
    print("Redis is not running")



def clear_history(session_id):
    redis_client.delete(f"chat:{session_id}")

def save_message(session_id, role, content):
    if not redis_client.ping():
        print("Redis is not running")
        return
    redis_client.rpush(
        f"chat:{session_id}",
        json.dumps(
            {
                "role": role,
                "content": content,
            }
        ),
    )
    
    
def load_chat_history(session_id):
    history = []

    messages = redis_client.lrange(f"chat:{session_id}", 0, -1)

    for msg in messages:
        msg = json.loads(msg)

        if msg["role"] == "human":
            history.append(
                HumanMessage(content=msg["content"])
            )
        else:
            history.append(
                AIMessage(content=msg["content"])
            )

    return history


# rag_chain = create_retrieval_chain(
#     retriever=retriever,
#     combine_documents_chain=document_chain)










contextualize_q_system_prompt="""   
Given a chat history and the latest user question which might 
reference context in the chat history, formulate a standalone question 
which can be understand without the chat history. Do not answer the question, 
just reformulate it if  needed and otherwise return it as is.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)
qa_system_prompt = """
You are an assistant for question answering tasks.

Use the following pieces of retrieved context to answer the question.

Context:
{context}

If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)

conversation_rag_chain  = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)   


session_id = str(uuid.uuid4())

while True:

    chat_history = load_chat_history(session_id)
    print("Chat History : ...............",end="\n")
    print(chat_history)
    print("----------------------------------------- ",end="\n\n")

    user_input = input("Qust: Enter your question: ")
    if user_input.lower() == "exit":
        print("Chat History : ...............",end="\n")
        print(chat_history) 
        clear_history(session_id)

        break
    result = conversation_rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input})
    print("----------------------------------------- ",end="\n\n")
    print(f"Answer: {result['answer']}")
    save_message(
        session_id,
        "human",
        user_input,
    )

    save_message(
        session_id,
        "ai",
        result["answer"],
    )
    print("----------------------------------------- ",end="\n\n")


