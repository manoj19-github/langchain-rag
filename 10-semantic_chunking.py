from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import Annotated,List
from typing_extensions import TypedDict
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint,HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

model = SentenceTransformer('all-MiniLM-L6-v2')

text=""""
Langchain is a framework for developing applications powered by LLMs (Large Language Models). It provides a set of tools and abstractions for building applications with the help of LLMs.

Langchain is built on top of the Langchain Python package, which provides a set of tools and abstractions for building applications with the help of LLMs. It provides a set of tools and abstractions for building applications with the help of LLMs.
You can create chain, agent, memory and retrievers. 
The Eiffel Tower is a famous landmark in Paris, France. It was built in 1889 for the World's Fair and was the tallest man-made structure in the world until 1973.
France is a country in Western Europe. It is located in the Northern Hemisphere and borders several countries, including Belgium, Germany, Italy, Luxembourg, Monaco, and Switzerland.
"""

# Step 1: Split the text into sentences
sentences = [s.strip() for s in text.split('\n') if s.strip()]
embeddings = model.encode(sentences)

# Step 2: Compute the cosine similarity between the embeddings of the sentences
threshold = 0.7
chunks=[]
current_chunk=[sentences[0]]

for i in range(1,len(sentences)):
    sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])
    if sim >= threshold:
        current_chunk.append(sentences[i])
    else:
        chunks.append(" ".join(current_chunk))
        current_chunk=[sentences[i]]
chunks.append(" ".join(current_chunk))



class ThresholdSemnaticChunker:
    def __init__(self, threshold: float = 0.7,model_name:str="all-MiniLM-L6-v2"):
        self.threshold = threshold
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    def split(self,text:str):
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        embeddings = self.model.encode(sentences)
        chunks=[]
        current_chunk=[sentences[0]]
        for i in range(1,len(sentences)):
            sim = cosine_similarity(
                    [embeddings[i-1]],
                    [embeddings[i]]
                )[0][0]
            if sim >= self.threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk=[sentences[i]]
        chunks.append(" ".join(current_chunk))
        return chunks
    def split_document(self,docs):
        result = []
        for doc in docs:
            for chunk in self.split(doc.page_content):
                result.append(
                     Document(
                    page_content=chunk,
                    metadata=doc.metadata
                )
                )
        return result
    
doc = Document(page_content=text)
chunker = ThresholdSemnaticChunker(threshold=0.7,model_name="all-MiniLM-L6-v2")
chunks = chunker.split_document([doc])

embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embed_model,
    persist_directory="./chroma_db",
    collection_name="rag_collection",
)

llm = ChatOllama(
    model="qwen3:4b-instruct",
    temperature=0.4
)   

retriever = vector_store.as_retriever(search_kwargs={"k":3})

#   LCEL  chain with retrieval 


template = """
Answer the question based on the following context: 
{context}
Question: {question}
"""
prompt = PromptTemplate.from_template(template)
rag_chain = (RunnableMap(
        {
            "context": lambda x: "\n\n".join(
                          doc.page_content
                        for doc in retriever.invoke(x["question"])
            ),
            "question":lambda x: x["question"]
        }
    ) | prompt | llm | StrOutputParser()
)


question = input("Enter your question: ")
result = rag_chain.invoke({"question":question})
print(result)
