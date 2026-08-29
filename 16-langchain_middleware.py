import os
import time
import hashlib
from operator import itemgetter
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage,SystemMessage
from langchain.agents.middleware import SummarizationMiddleware


config = {
    "configurable": {
        "thread_id": "test-1"
    }
}
model = ChatOllama(
    model="qwen3:4b-instruct",
    temperature=0.4
)

agent = create_agent(
    model = ChatOllama(
        model="qwen3:4b-instruct",
        temperature=0.4
    ),
    tools=[],
    checkpointer=InMemorySaver(),
    middleware=[
       SummarizationMiddleware(
    trigger=("messages", 10),
    model=model,
    keep=("messages", 4)
)

    ]
)




questions = [
    "what is 2*2",
    "what is 2*4",
    "what is 4*4",
    "what is 57/8",
    "what is 4*4",
    "what is 2*4",
    "what is 2*4",
    "what is 15-3",
    "what is 2*4",
    "what is 34+53",
    "what is 2*4",
    "what is 2*4",
]

for q in questions:
    response = agent.invoke({"messages":[HumanMessage(content=q)]},config=config)
    print("response : ",response)
    print("len messsages : ",len(response["messages"]))
    print("=============================")