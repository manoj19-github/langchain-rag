import os
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
from operator import itemgetter
# 1. Create documents

from langchain_community.document_loaders import TextLoader,PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyMuPDFLoader("data/pdf/Plutus-IAS_Ancient-Indian-History-Dr.-Jitendra-Kumar-1.pdf")
raw_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(raw_docs)
print(f"Loaded and split into {len(documents)} documents.")

# 2. Create Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# 3. Create Chroma Vectorstore
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="hrms_documents",
    persist_directory="./chroma_db_hrms"
)

# 4. Dense MMR Retriever
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 40,
        "lambda_mult": 0.6
    }
)

# 5. Sparse BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10


# def hybrid_mmr_retrieve(query):
#     dense_docs = mmr_retriever.invoke(query)
#     sparse_docs = bm25_retriever.invoke(query)

#     combined = []
#     seen = set()

#     for doc in dense_docs + sparse_docs:
#         key = doc.page_content.strip()
#         if key not in seen:
#             combined.append(doc)
#             seen.add(key)

#     return combined
# 6. Hybrid Retrieval with Deduplication
hybrid_retriever = EnsembleRetriever(
    retrievers=[mmr_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

# 7. Initialize Cross-Encoder Reranker
# Using ms-marco-MiniLM-L6-v2 as discussed in the context
print("Loading CrossEncoder Reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def rerank(query, candidates, final_k=5):
    pairs = [
        (query, doc.page_content)
        for doc in candidates
    ]
    
    # Score pairs
    scores = reranker.predict(pairs, batch_size=16)
    
    # Sort by score
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    return ranked[:final_k]


if __name__ == "__main__":
    query = "Write a short note about Cholas."
    
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80 + "\n")
    
    # Step A: Hybrid MMR Retrieval
    candidates = hybrid_retriever.invoke(query)
    
    print(f"Retrieved {len(candidates)} deduplicated candidates from Chroma MMR and BM25.\n")
    print("CANDIDATES BEFORE RERANKING:")
    for i, doc in enumerate(candidates):
        print(f"  [{i+1}] {doc.page_content.strip()} (Source: {doc.metadata.get('source')})")
        
    print("\n" + "=" * 80)
    
    # Step B: Cross-Encoder Reranking
    results = rerank(query, candidates, final_k=3)
    
    print("FINAL RERANKED RESULTS:")
    print("=" * 80)
    for doc, score in results:
        print(f"Score: {score:.4f} | Source: {doc.metadata.get('source')}")
        print(doc.page_content.strip())
        print("-" * 80)
        
    print("\n" + "=" * 80)
    print("RUNNING LLM GENERATION WITH LCEL RAG CHAIN...")
    print("=" * 80 + "\n")
    
    # Define LLM and Prompt
    llm = ChatOllama(model="qwen3:4b-instruct", temperature=0.4)
    
    template = """
    Answer the question based on the following context.
    
    Context:
    {context}
    
    Question:
    {input}
    """
    prompt = PromptTemplate.from_template(template)
    
    # Helper to map input to our existing retrieve and rerank flow
    def run_retrieval_flow(query_str):
        candidates_list = hybrid_retriever.invoke(query_str)
        scored_docs = rerank(query_str, candidates_list, final_k=3)
        # Extract just the Document objects from the (doc, score) tuples
        return [doc for doc, score in scored_docs]
        
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # The Complete LCEL Chain
    rag_chain = (
        {
            "context": (
                itemgetter("input")
                | RunnableLambda(run_retrieval_flow)
                | RunnableLambda(format_docs)
            ),
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Generate the final answer
    final_answer = rag_chain.invoke({"input": query})
    print("FINAL ANSWER:")
    print(final_answer)
