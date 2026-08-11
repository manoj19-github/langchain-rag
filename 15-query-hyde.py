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

# ==================================================
# CONFIGURATION
# ==================================================
MMR_K = 10
MMR_FETCH_K = 40
MMR_LAMBDA = 0.6
BM25_K = 10
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
RERANK_TOP_K = 6
RERANK_BATCH_SIZE = 16
HYDE_TEMPERATURE = 0.4
DEBUG_LOGGING = True

# ==================================================
# 1. SETUP DATA AND RETRIEVERS
# ==================================================
loader = PyMuPDFLoader("data/pdf/Plutus-IAS_Ancient-Indian-History-Dr.-Jitendra-Kumar-1.pdf")
raw_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(raw_docs)
if DEBUG_LOGGING:
    print(f"Loaded and split into {len(documents)} documents.")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="hrms_documents",
    persist_directory="./chroma_db_hrms"
)

# Dense MMR Retriever
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": MMR_K,
        "fetch_k": MMR_FETCH_K,
        "lambda_mult": MMR_LAMBDA
    }
)

# Sparse BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = BM25_K

if DEBUG_LOGGING:
    print("Loading CrossEncoder Reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


# ==================================================
# 2. HyDE MODEL
# ==================================================
hyde_llm = ChatOllama(
    model="qwen3:4b-instruct",
    temperature=HYDE_TEMPERATURE
)

HYDE_PROMPT = """You are a retrieval enhancement component in a Retrieval-Augmented Generation system.

Your task is to generate a hypothetical document that could contain the information needed to answer the user's question.

This document will NOT be used as the final answer.
It will only be converted into an embedding and used for semantic retrieval from a knowledge base.

Instructions:
1. Generate a concise hypothetical passage relevant to the user's question.
2. Include terminology, concepts, entities, and relationships that are likely to appear in relevant source documents.
3. Focus on the information needed to answer the question.
4. Do not discuss the retrieval process.
5. Do not mention that the document is hypothetical.
6. Do not use Markdown.
7. Do not provide a list of alternative queries.
8. Do not decompose the question into multiple questions.
9. Do not generate multiple documents.
10. Do not add irrelevant information.
11. The output should resemble a natural knowledge-base passage.
12. The generated text is only a retrieval aid and may contain uncertain information.

User Question:
{query}

Generate the hypothetical document:
"""

hyde_template = PromptTemplate.from_template(HYDE_PROMPT)
hyde_chain = hyde_template | hyde_llm | StrOutputParser()

# ==================================================
# HyDE GENERATION FUNCTION
# ==================================================
def generate_hypothetical_document(query: str) -> str:
    try:
        doc = hyde_chain.invoke({"query": query})
        doc = doc.strip()
        if not doc:
            return query
        return doc
    except Exception as e:
        if DEBUG_LOGGING:
            print(f"HyDE generation failed: {e}. Falling back to original query.")
        return query

# ==================================================
# DOCUMENT DEDUPLICATION
# ==================================================
def document_key(doc: Document) -> str:
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    content = doc.page_content.strip()
    raw_key = f"{source}|{page}|{content}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

# ==================================================
# HYBRID RETRIEVAL & MERGE (REPLACES ENSEMBLE RETRIEVER)
# ==================================================
def retrieve_dense_with_hyde(query_str: str, hypothetical_document: str) -> List[Document]:
    docs1 = mmr_retriever.invoke(hypothetical_document)
    docs2 = mmr_retriever.invoke(query_str)
    return docs1 + docs2

def retrieve_sparse(original_query: str) -> List[Document]:
    return bm25_retriever.invoke(original_query)

def merge_hybrid_results(dense_documents: List[Document], sparse_documents: List[Document]) -> List[Document]:
    # We will manually merge using Reciprocal Rank Fusion weighted by DENSE_WEIGHT and SPARSE_WEIGHT
    # to emulate EnsembleRetriever behavior but with different source queries.
    
    doc_scores = {}
    doc_map = {}
    
    # Process Dense Documents
    for rank, doc in enumerate(dense_documents):
        key = document_key(doc)
        if key not in doc_map:
            doc_map[key] = doc
            doc_scores[key] = 0.0
        # RRF formula: 1 / (rank + 60)
        score = 1.0 / (rank + 60)
        doc_scores[key] += DENSE_WEIGHT * score
        
    # Process Sparse Documents
    for rank, doc in enumerate(sparse_documents):
        key = document_key(doc)
        if key not in doc_map:
            doc_map[key] = doc
            doc_scores[key] = 0.0
        score = 1.0 / (rank + 60)
        doc_scores[key] += SPARSE_WEIGHT * score
        
    # Sort by the fused score
    sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys]

def deduplicate_documents(documents: List[Document]) -> List[Document]:
    seen = set()
    deduped = []
    for doc in documents:
        key = document_key(doc)
        if key not in seen:
            seen.add(key)
            deduped.append(doc)
    return deduped

# ==================================================
# RERANKING FUNCTION
# ==================================================
def rerank(query: str, candidates: List[Document], final_k: int = RERANK_TOP_K) -> List[tuple[Document, float]]:
    if not candidates:
        return []
        
    pairs = [
        (query, doc.page_content)
        for doc in candidates
    ]
    
    try:
        # Score pairs
        scores = reranker.predict(pairs, batch_size=RERANK_BATCH_SIZE)
        
        # Sort by score
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked[:final_k]
    except Exception as e:
        if DEBUG_LOGGING:
            print(f"CrossEncoder reranking failed: {e}")
        return [(c, 0.0) for c in candidates[:final_k]]


# ==================================================
# COMPLETE RAG RETRIEVAL FLOW
# ==================================================
rag_llm = ChatOllama(model="qwen3:4b-instruct", temperature=0.4)

rag_template = """Answer the question using only the provided context.

Do not treat the hypothetical HyDE document as evidence.
If the context does not contain enough information to answer the question, clearly say that the available context is insufficient.

Context:
{context}

Question:
{input}
"""
rag_prompt = PromptTemplate.from_template(rag_template)

def run_retrieval_flow(query_str: str) -> List[Document]:
    t0 = time.time()
    
    if DEBUG_LOGGING:
        print("\n" + "=" * 60)
        print("ORIGINAL QUERY")
        print("=" * 60)
        print(f"{query_str}\n")
        
        print("=" * 60)
        print("HYDE")
        print("=" * 60)
        print("Generating hypothetical document...\n")
        
    # Step 1: Generate HyDE document
    t1 = time.time()
    hypothetical_document = generate_hypothetical_document(query_str)
    hyde_time = time.time() - t1
    
    if DEBUG_LOGGING:
        print("-" * 60)
        print("HYPOTHETICAL DOCUMENT")
        print("-" * 60)
        print(hypothetical_document.encode('ascii', 'ignore').decode('ascii'))
        print("-" * 60 + "\n")
        
    # Step 2: Dense MMR using HyDE and Original Query
    t2 = time.time()
    dense_documents = retrieve_dense_with_hyde(query_str, hypothetical_document)
    dense_time = time.time() - t2
    
    if DEBUG_LOGGING:
        print("=" * 60)
        print("DENSE MMR RETRIEVAL")
        print("=" * 60)
        print("HyDE-based dense retrieval completed.")
        print(f"Retrieved: {len(dense_documents)} documents\n")

    # Step 3: Sparse BM25 using Original Query
    t3 = time.time()
    sparse_documents = retrieve_sparse(query_str)
    sparse_time = time.time() - t3
    
    if DEBUG_LOGGING:
        print("=" * 60)
        print("BM25 RETRIEVAL")
        print("=" * 60)
        print("Original-query BM25 retrieval completed.")
        print(f"Retrieved: {len(sparse_documents)} documents\n")
        
    # Step 4: Hybrid Merge
    t4 = time.time()
    merged_candidates = merge_hybrid_results(dense_documents, sparse_documents)
    
    # Step 5: Deduplicate
    candidates = deduplicate_documents(merged_candidates)
    merge_time = time.time() - t4
    
    if DEBUG_LOGGING:
        print("=" * 60)
        print("HYBRID MERGE")
        print("=" * 60)
        print(f"Dense documents: {len(dense_documents)}")
        print(f"BM25 documents: {len(sparse_documents)}")
        print(f"Before deduplication: {len(dense_documents) + len(sparse_documents)}")
        print(f"After deduplication: {len(candidates)}\n")
        
    # Step 6: Global CrossEncoder
    t5 = time.time()
    scored_docs = rerank(query_str, candidates, final_k=RERANK_TOP_K)
    rerank_time = time.time() - t5
    
    if DEBUG_LOGGING:
        print("=" * 60)
        print("CROSSENCODER RERANKING")
        print("=" * 60)
        print("Ranking candidates against ORIGINAL QUERY...\n")
        for i, (doc, score) in enumerate(scored_docs):
            print(f"[{i+1}] Score: {score:.4f}")
            
        print("\n" + "=" * 60)
        print("FINAL CONTEXT")
        print("=" * 60)
        print(f"Top K documents: {len(scored_docs)}\n")
        
        print("=" * 60)
        print("LATENCY TIMING")
        print("=" * 60)
        print(f"HyDE generation:       {hyde_time:.2f} sec")
        print(f"Dense MMR retrieval:   {dense_time:.2f} sec")
        print(f"BM25 retrieval:        {sparse_time:.2f} sec")
        print(f"Merge/Dedup:           {merge_time:.2f} sec")
        print(f"CrossEncoder:          {rerank_time:.2f} sec")
        total_time = time.time() - t0
        print(f"Total retrieval time:  {total_time:.2f} sec\n")

    # Step 7: Return documents only
    return [doc for doc, score in scored_docs]

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": (
            itemgetter("input")
            | RunnableLambda(run_retrieval_flow)
            | RunnableLambda(format_docs)
        ),
        "input": itemgetter("input")
    }
    | rag_prompt
    | rag_llm
    | StrOutputParser()
)

# ==================================================
# TESTING
# ==================================================
if __name__ == "__main__":
    test_queries = [
        "Who was Rajaraja Chola I?", # Simple factual query
        "What were the major features of Chola administration?", # Analytical
        "Explain the military expansion of the Cholas.", # Broad
        "Rajaraja Chola I" # Exact entity
    ]
    
    # Test just one query to demonstrate the pipeline
    test_query = test_queries[1]
    
    print("\n\n" + "*" * 80)
    print("STARTING RAG PIPELINE WITH HYDE")
    print("*" * 80)
    
    docs = run_retrieval_flow(test_query)
    print("\n" + "=" * 60)
    print("FINAL CONTEXT CONTENT")
    print("=" * 60)
    print(format_docs(docs).encode('ascii', 'ignore').decode('ascii'))
    print("=" * 60 + "\n")
    
    t_start = time.time()
    final_answer = rag_chain.invoke({"input": test_query})
    llm_time = time.time() - t_start
    
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(final_answer)
    print("=" * 60)
    print(f"Total End-to-End Latency: {llm_time:.2f} sec\n")
