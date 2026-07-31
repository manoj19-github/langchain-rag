
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import EnsembleRetriever
from operator import itemgetter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.chains import create_retrieval_chain
from langchain_core.runnables import RunnableLambda
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama

import re
import os
docs = [
    Document(page_content="Langchain helps to build LLM applications", metadata={"source": "my-source"}),
    Document(page_content="Pinecode in a vector database for semantic search", metadata={"source": "my-source"}),
    Document(page_content="The Eiffel Tower is a famous landmark in Paris, France", metadata={"source": "my-source"}),
    Document(page_content="Paris is the capital and most populous city of France", metadata={"source": "my-source"}),
    Document(page_content="France is a country in Western Europe", metadata={"source": "my-source"}),
]

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    
)
dense_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma_db",
    collection_name="rag_collection",
)

dense_retriever = dense_store.as_retriever(search_kwargs={"k":10})

sparse_retriever = BM25Retriever.from_documents(docs)
sparse_retriever.k=10

hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3],   # ✅
)

template = """
Answer the question based on the following context.

Context:
{context}

Question:
{input}
"""
prompt = PromptTemplate.from_template(template)
# rag_chain = (EnsembleRetriever(
#         retrievers=[dense_retriever],
#         combine_documents_chain=None,
#         verbose=True
#     ) | prompt | HumanMessage(content="Answer the question based on the following context: {context}\nQuestion: {question}")
# )

question = """ How can i use langchain to build an application with memory and tools?  """
result = hybrid_retriever.invoke(question)

llm = ChatOllama(
    model="qwen3:4b-instruct",
    temperature=0.4
)


# combine_docs_chain = create_stuff_documents_chain(
#     llm,
#     prompt,
# )

# rag_chain = create_retrieval_chain(
#     hybrid_retriever,
#     combine_docs_chain,
# )



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def formatted_docs(docs):
    doc_lines = [f"{i+1}. {doc.page_content}" for i,doc in enumerate(docs)]
    return "\n".join(doc_lines)

# -------------------------------------------------
# LCEL RAG Chain
# -------------------------------------------------



reranking_prompt = PromptTemplate.from_template(
    """  
    You are helpfull assistant. Your task is to rank the following documents from most to least relevant 
    
    User Question: {input}
    
    Documents:
    {documents}
    
    Instructions:
    - Think about the relevance of each document to the user's question.
    - Return a list of document indices in ranked order, starting from the most relevant.
    
    Output format: comma-separated document indices (e.g "2,1,3,0")
    """
    
    
)


rerank_chain = (
    reranking_prompt
    | llm
    | StrOutputParser()
)



def rerank_documents(inputs):
    question = inputs["input"]

    # -----------------------------
    # Retrieve candidate documents
    # -----------------------------
    print("Question:", question)
    print("=" * 70)
    docs = hybrid_retriever.invoke(question)

    print("\n" + "=" * 70)
    print("DOCUMENTS BEFORE RERANKING")
    print("=" * 70)

    for i, doc in enumerate(docs):
        print(f"[{i}] {doc.page_content}")

    formatted = "\n\n".join(
        f"{i}. {doc.page_content}"
        for i, doc in enumerate(docs)
    )

    # -----------------------------
    # LLM Reranking
    # -----------------------------
    ranking = rerank_chain.invoke(
        {
            "input": question,
            "documents": formatted
        }
    )

    print("\nLLM Ranking Output:", ranking)

    # Extract document indices
    indices = [
        int(x)
        for x in re.findall(r"\d+", ranking)
        if int(x) < len(docs)
    ]

    # Remove duplicate indices
    seen = set()
    ordered = []
    for idx in indices:
        if idx not in seen:
            ordered.append(docs[idx])
            seen.add(idx)

    # Fallback
    if not ordered:
        ordered = docs

    print("\n" + "=" * 70)
    print("DOCUMENTS AFTER RERANKING")
    print("=" * 70)

    for i, doc in enumerate(ordered):
        print(f"[{i}] {doc.page_content}")

    print("=" * 70)

    return ordered[:9]


rag_chain = (
    {
        "context": (
            itemgetter("input")
            | RunnableLambda(
                lambda q: {
                    "input": q
                }
            )
            | RunnableLambda(rerank_documents)
            | RunnableLambda(formatted_docs)
        ),
        "input": itemgetter("input")
    }
    | prompt
    | llm
    | StrOutputParser()
)



response = rag_chain.invoke(
    {
        "input": question
    }
)
print(response)

    