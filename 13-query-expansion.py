import os
import json
import hashlib
from operator import itemgetter

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==================================================
# 21. CONFIGURATION
# ==================================================
NUM_EXPANDED_QUERIES = 3
MMR_K = 10
MMR_FETCH_K = 40
MMR_LAMBDA = 0.6
BM25_K = 10
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
RERANK_TOP_K = 3
RERANK_BATCH_SIZE = 16
QUERY_EXPANSION_TEMPERATURE = 0.2
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

# Keep existing MMR configuration
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": MMR_K,
        "fetch_k": MMR_FETCH_K,
        "lambda_mult": MMR_LAMBDA
    }
)

# Keep existing BM25 configuration
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = BM25_K

# Keep existing Ensemble Retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[mmr_retriever, bm25_retriever],
    weights=[DENSE_WEIGHT, SPARSE_WEIGHT]
)

if DEBUG_LOGGING:
    print("Loading CrossEncoder Reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


# ==================================================
# 2. QUERY EXPANSION MODEL
# ==================================================
query_expansion_llm = ChatOllama(
    model="qwen3:4b-instruct",
    temperature=QUERY_EXPANSION_TEMPERATURE
)

QUERY_EXPANSION_PROMPT = """You are a query expansion component in a RAG retrieval system.
Your task is to generate alternative search queries for the user's question.

Original question:
{query}

Generate {num_queries} diverse search queries that improve document retrieval.

Requirements:
1. Preserve the original intent.
2. Cover different relevant aspects of the question.
3. Expand important entities and terminology.
4. Include useful synonyms or related terminology when appropriate.
5. Create queries suitable for both semantic retrieval and keyword retrieval.
6. Do not answer the question.
7. Do not introduce unrelated information.
8. Avoid duplicate queries.
9. Keep each query concise.
10. Return ONLY valid JSON.

Required format:
{{
    "queries": [
        "query 1",
        "query 2",
        "query 3"
    ]
}}
"""

query_expansion_template = PromptTemplate.from_template(QUERY_EXPANSION_PROMPT)
query_expansion_chain = query_expansion_template | query_expansion_llm | StrOutputParser()

# ==================================================
# 4. QUERY EXPANSION FUNCTION
# ==================================================
def expand_query(query: str, num_queries: int = NUM_EXPANDED_QUERIES) -> list[str]:
    expanded_queries = [query.strip()]
    
    try:
        response = query_expansion_chain.invoke({"query": query, "num_queries": num_queries})
        # Strip potential markdown formatting
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        data = json.loads(response)
        if "queries" in data and isinstance(data["queries"], list):
            for q in data["queries"]:
                q = q.strip()
                if q and q.lower() not in [eq.lower() for eq in expanded_queries]:
                    expanded_queries.append(q)
    except Exception as e:
        if DEBUG_LOGGING:
            print(f"Query expansion parsing failed, falling back to original query: {e}")
        # Fallback parsing strategy if JSON fails (extract lines)
        try:
            lines = response.split("\n")
            for line in lines:
                clean_line = line.strip()
                # Remove common numbering 1. 2. - * etc
                if clean_line and clean_line[0].isdigit() and (clean_line[1] == "." or clean_line[1] == ")"):
                    clean_line = clean_line[2:].strip()
                if clean_line.startswith("-") or clean_line.startswith("*"):
                    clean_line = clean_line[1:].strip()
                
                if clean_line and clean_line.lower() not in [eq.lower() for eq in expanded_queries] and len(clean_line) > 5 and not clean_line.startswith("{") and not clean_line.startswith("["):
                    expanded_queries.append(clean_line)
        except Exception:
            pass # Keep only the original query
            
    # Limit to original query + num_queries expansions
    return expanded_queries[:num_queries + 1]

# ==================================================
# 10. DOCUMENT DEDUPLICATION
# ==================================================
def document_key(doc: Document) -> str:
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    content = doc.page_content.strip()

    raw_key = f"{source}|{page}|{content}"

    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# ==================================================
# 9. MULTI-QUERY RETRIEVAL FUNCTION
# ==================================================
def retrieve_with_query_expansion(original_query: str, num_queries: int = NUM_EXPANDED_QUERIES) -> list[Document]:
    expanded_queries = expand_query(original_query, num_queries)
    
    if DEBUG_LOGGING:
        print("\n" + "=" * 50)
        print("QUERY EXPANSION")
        print("=" * 50)
        print(f"ORIGINAL QUERY:\n{original_query}\n")
        print("EXPANDED QUERIES:")
        for i, q in enumerate(expanded_queries):
            print(f"[{i}] {q}")

    all_candidates = {}
    
    if DEBUG_LOGGING:
        print("\n" + "=" * 50)
        print("HYBRID RETRIEVAL")
        print("=" * 50)

    for i, q in enumerate(expanded_queries):
        docs = hybrid_retriever.invoke(q)
        
        if DEBUG_LOGGING:
            print(f"Query [{i}] -> {len(docs)} candidates")
            
        for doc in docs:
            key = document_key(doc)
            if key not in all_candidates:
                # Add metadata to track provenance
                doc.metadata["retrieval_query"] = q
                doc.metadata["query_index"] = i
                doc.metadata["matched_queries"] = [q]
                all_candidates[key] = doc
            else:
                if q not in all_candidates[key].metadata.get("matched_queries", []):
                    all_candidates[key].metadata.setdefault("matched_queries", []).append(q)
                    
    global_candidates = list(all_candidates.values())
    
    if DEBUG_LOGGING:
        total_retrieved = sum(len(hybrid_retriever.invoke(q)) for q in expanded_queries)
        print(f"\nTotal candidates before deduplication: {total_retrieved}")
        print(f"Candidates after deduplication: {len(global_candidates)}")
        
    return global_candidates

# ==================================================
# 13. RERANKING FUNCTION
# ==================================================
def rerank(query: str, candidates: list[Document], final_k: int = RERANK_TOP_K):
    if not candidates:
        return []
        
    pairs = [
        (query, doc.page_content)
        for doc in candidates
    ]
    
    # Score pairs
    scores = reranker.predict(pairs, batch_size=RERANK_BATCH_SIZE)
    
    # Sort by score
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    return ranked[:final_k]


# ==================================================
# 16. UPDATED LCEL ARCHITECTURE
# ==================================================
rag_llm = ChatOllama(model="qwen3:4b-instruct", temperature=0.4)

rag_template = """Answer the question based only on the following context.

Context:
{context}

Question:
{input}

Rules:
- Use the supplied context.
- Do not invent facts.
- If the context does not contain enough information, clearly state that.
- Answer the original question directly.
- Keep the response concise when the user requests a short answer.
"""
rag_prompt = PromptTemplate.from_template(rag_template)

def run_retrieval_flow(query_str: str) -> list[Document]:
    # 1-3. Query expansion, Multi-query hybrid retrieval, Global deduplication
    global_candidates = retrieve_with_query_expansion(query_str, NUM_EXPANDED_QUERIES)
    
    # 4. CrossEncoder reranking (using ORIGINAL QUERY)
    scored_docs = rerank(query_str, global_candidates, final_k=RERANK_TOP_K)
    
    if DEBUG_LOGGING:
        print("\n" + "=" * 50)
        print("CROSS-ENCODER RERANKING")
        print("=" * 50)
        print("Top results:")
        for i, (doc, score) in enumerate(scored_docs):
            print(f"[{i+1}] Score: {score:.4f} | Q-Index: {doc.metadata.get('query_index')} | Queries: {len(doc.metadata.get('matched_queries', []))}")
            
        print("\n" + "=" * 50)
        print("FINAL CONTEXT")
        print("=" * 50)
        print(f"{len(scored_docs)} documents sent to LLM.")
        
    # 5. Return final Document objects
    return [doc for doc, score in scored_docs]

def format_docs(docs: list[Document]) -> str:
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
# 28. TESTING
# ==================================================
if __name__ == "__main__":
    test_queries = [
        "Write a short note about Cholas.",
        "What were the major achievements of the Mauryas?",
        "Who was Ashoka?",
        "Explain ancient Indian administration.",
        "What was the role of temples during the Chola period?",
        "Chola administration",
        "Explain the political and economic structure of the Chola Empire."
    ]
    
    # Test just one query to demonstrate the pipeline
    test_query = test_queries[0]
    
    print("\n\n" + "*" * 80)
    print("STARTING RAG PIPELINE WITH QUERY EXPANSION")
    print("*" * 80)
    
    final_answer = rag_chain.invoke({"input": test_query})
    
    print("\n" + "=" * 50)
    print("FINAL ANSWER")
    print("=" * 50)
    print(final_answer)
    print("=" * 50 + "\n")
