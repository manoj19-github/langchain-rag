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
MAX_SUB_QUERIES = 5
MMR_K = 10
MMR_FETCH_K = 40
MMR_LAMBDA = 0.6
BM25_K = 10
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
RERANK_TOP_K = 3
RERANK_BATCH_SIZE = 16
QUERY_DECOMPOSITION_TEMPERATURE = 0.2
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
# 2. QUERY DECOMPOSITION MODEL
# ==================================================
query_decomposition_llm = ChatOllama(
    model="qwen3:4b-instruct",
    temperature=QUERY_DECOMPOSITION_TEMPERATURE
)

QUERY_DECOMPOSITION_PROMPT = """You are a query decomposition component in a Retrieval-Augmented Generation system.

Your task is to analyze the user's question and break it into smaller, independent sub-questions that can be retrieved separately from a knowledge base.

Rules:
1. Preserve the original user's intent.
2. Do not answer the question.
3. Do not generate alternative phrasings of the same question.
4. Generate independent information requirements.
5. Each sub-question must be understandable on its own.
6. Each sub-question must contain enough context to be independently searchable.
7. Do not create unnecessary sub-questions.
8. For a simple factual question, return only one sub-question.
9. For a complex question, generate multiple focused sub-questions.
10. Do not introduce information that is not implied by the original question.
11. Keep each sub-question concise and retrieval-friendly.
12. Return valid JSON only.

User Question:
{query}

Return:
{{
  "sub_queries": [
    "..."
  ]
}}
"""

query_decomposition_template = PromptTemplate.from_template(QUERY_DECOMPOSITION_PROMPT)
query_decomposition_chain = query_decomposition_template | query_decomposition_llm | StrOutputParser()

# ==================================================
# 4. QUERY DECOMPOSITION FUNCTION
# ==================================================
def decompose_query(query: str, max_sub_queries: int = MAX_SUB_QUERIES) -> list[str]:
    sub_queries = []
    
    try:
        response = query_decomposition_chain.invoke({"query": query})
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
        if "sub_queries" in data and isinstance(data["sub_queries"], list):
            for q in data["sub_queries"]:
                q = q.strip()
                if q and q.lower() not in [eq.lower() for eq in sub_queries]:
                    sub_queries.append(q)
    except Exception as e:
        if DEBUG_LOGGING:
            print(f"Query decomposition parsing failed, falling back: {e}")
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
                
                if clean_line and clean_line.lower() not in [eq.lower() for eq in sub_queries] and len(clean_line) > 5 and not clean_line.startswith("{") and not clean_line.startswith("[") and not clean_line.startswith('"'):
                    sub_queries.append(clean_line)
        except Exception:
            pass
            
    if not sub_queries:
        return [query.strip()]
        
    return sub_queries[:max_sub_queries]

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
def retrieve_from_sub_queries(original_query: str, sub_queries: list[str]) -> list[Document]:
    all_candidates = {}
    
    if DEBUG_LOGGING:
        print("\n" + "=" * 60)
        print("RETRIEVAL")
        print("=" * 60)

    for i, q in enumerate(sub_queries):
        try:
            docs = hybrid_retriever.invoke(q)
            
            if DEBUG_LOGGING:
                print(f"Sub-query {i+1}:")
                print(f"Retrieved: {len(docs)} documents\n")
                
            for doc in docs:
                # Create a copy so we don't modify the object in Chroma's cache
                new_doc = Document(page_content=doc.page_content, metadata=doc.metadata.copy())
                key = document_key(new_doc)
                
                if key not in all_candidates:
                    # Add metadata to track provenance
                    new_doc.metadata["retrieval_sub_query"] = q
                    new_doc.metadata["matched_sub_queries"] = [q]
                    all_candidates[key] = new_doc
                else:
                    if q not in all_candidates[key].metadata.get("matched_sub_queries", []):
                        all_candidates[key].metadata.setdefault("matched_sub_queries", []).append(q)
        except Exception as e:
            if DEBUG_LOGGING:
                print(f"Retrieval for sub-query '{q}' failed: {e}")
                
    # Fallback if all sub-queries failed
    if not all_candidates:
        if DEBUG_LOGGING:
            print("All sub-queries failed, falling back to original query.")
        docs = hybrid_retriever.invoke(original_query)
        for doc in docs:
            new_doc = Document(page_content=doc.page_content, metadata=doc.metadata.copy())
            key = document_key(new_doc)
            new_doc.metadata["retrieval_sub_query"] = original_query
            new_doc.metadata["matched_sub_queries"] = [original_query]
            all_candidates[key] = new_doc
            
    global_candidates = list(all_candidates.values())
    
    if DEBUG_LOGGING:
        print("=" * 60)
        print("MERGED CANDIDATES")
        print("=" * 60)
        total_retrieved = sum(len(c.metadata.get("matched_sub_queries", [])) for c in global_candidates) # approximate before deduplication count
        print(f"Before deduplication (approx): {total_retrieved}")
        print(f"After deduplication: {len(global_candidates)}")
        
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
        # Fallback if reranking fails, just return top K as they were
        return [(c, 0.0) for c in candidates[:final_k]]


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
    # 1. Query decomposition
    sub_queries = decompose_query(query_str, MAX_SUB_QUERIES)
    
    if DEBUG_LOGGING:
        print("\n" + "=" * 60)
        print("ORIGINAL QUERY")
        print("=" * 60)
        print(f"{query_str}\n")
        
        print("=" * 60)
        print("DECOMPOSED SUB-QUERIES")
        print("=" * 60)
        for i, q in enumerate(sub_queries):
            print(f"[{i+1}] {q}")

    # 2. Multi-query hybrid retrieval and deduplication
    global_candidates = retrieve_from_sub_queries(query_str, sub_queries)
    
    # 3. Global CrossEncoder reranking (using ORIGINAL QUERY)
    scored_docs = rerank(query_str, global_candidates, final_k=RERANK_TOP_K)
    
    if DEBUG_LOGGING:
        print("\n" + "=" * 60)
        print("CROSSENCODER RERANKING")
        print("=" * 60)
        for i, (doc, score) in enumerate(scored_docs):
            print(f"Candidate {i+1} -> Score: {score:.4f} | Sub-queries: {len(doc.metadata.get('matched_sub_queries', []))}")
            
        print("\n" + "=" * 60)
        print("FINAL TOP-K DOCUMENTS")
        print("=" * 60)
        print(f"{len(scored_docs)} documents sent to LLM.")
        
    # 4. Return final Document objects
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
        "Who was Rajaraja Chola I?", # Simple factual query
        "Explain the Chola administration, economy and military system.", # Multi-part query
        "Compare Mauryan and Chola administration.", # Comparative query
        "Write a detailed note about the Chola dynasty." # Broad question
    ]
    
    # Test just one query to demonstrate the pipeline, let's use the comparative one for robust demonstration
    test_query = test_queries[2]
    
    print("\n\n" + "*" * 80)
    print("STARTING RAG PIPELINE WITH QUERY DECOMPOSITION")
    print("*" * 80)
    
    final_answer = rag_chain.invoke({"input": test_query})
    
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(final_answer)
    print("=" * 60 + "\n")
