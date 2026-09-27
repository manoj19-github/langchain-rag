import os
import warnings
from typing import List
from typing_extensions import TypedDict
from dotenv import load_dotenv

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

# If HF_TOKEN in environment is invalid/expired, ignore it for public models
if os.getenv("HF_TOKEN") == "santraai" or (os.getenv("HF_TOKEN") and not os.getenv("HF_TOKEN").startswith("hf_")):
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

# ---------------------------------------------------------------------------
# 1. Configuration & Embedding / Model Initialization
# ---------------------------------------------------------------------------
PDF_PATH = os.path.join("data", "pdf", "ccab2-geography.pdf")
CHROMA_PERSIST_DIR = "./chroma_db_geography"
COLLECTION_NAME = "geography_rag"

# Embedding Model (Cached locally or downloaded from HuggingFace without auth failure)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu", "token": False}
)

# LLM setup (using local Ollama with fallback)
llm = ChatOllama(
    model="qwen3:4b-instruct",
    temperature=0.2,
)

# ---------------------------------------------------------------------------
# 2. Vector Store Setup (Index once or load from disk)
# ---------------------------------------------------------------------------
def get_vector_store() -> Chroma:
    """Loads existing Chroma vector store or builds a new one from the PDF."""
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        print(f"[VectorStore] Loading existing ChromaDB from '{CHROMA_PERSIST_DIR}'...")
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
    else:
        print(f"[VectorStore] Ingesting and indexing PDF: {PDF_PATH}...")
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"PDF file not found at path: {PDF_PATH}")
        
        # Load PDF using PyMuPDFLoader
        loader = PyMuPDFLoader(PDF_PATH)
        documents = loader.load()
        print(f"[VectorStore] Loaded {len(documents)} pages from PDF.")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[VectorStore] Created {len(chunks)} text chunks.")

        # Create and persist vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=COLLECTION_NAME,
        )
        print("[VectorStore] ChromaDB index created and saved successfully.")

    return vector_store

# Initialize Vector Store & Retriever
vector_store = get_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 4})


# ---------------------------------------------------------------------------
# 3. LangGraph RAG State and Nodes Definition
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str


def retrieve(state: GraphState) -> dict:
    """Retrieve relevant documents for the given question."""
    question = state["question"]
    print(f"\n[Node: Retrieve] Searching relevant context for: '{question}'...")
    documents = retriever.invoke(question)
    print(f"[Node: Retrieve] Found {len(documents)} relevant chunks.")
    return {"documents": documents, "question": question}


def generate(state: GraphState) -> dict:
    """Generate an answer using the retrieved context and question."""
    question = state["question"]
    documents = state["documents"]
    print(f"[Node: Generate] Generating response from context with LLM...")

    # Format document contents with metadata references
    context_str = "\n\n".join([
        f"[Page {doc.metadata.get('page', 'N/A')}]:\n{doc.page_content}"
        for doc in documents
    ])

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert Geography assistant. Answer the question based ONLY on the provided context.\n"
            "If the answer cannot be found in the context, concisely state that you do not know.\n"
            "Include page references when applicable.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ])

    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": context_str, "question": question})

    return {"generation": generation, "documents": documents, "question": question}


# ---------------------------------------------------------------------------
# 4. Build and Compile the LangGraph Workflow
# ---------------------------------------------------------------------------
def build_rag_graph():
    workflow = StateGraph(GraphState)

    # Define Nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    # Define Edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile Graph
    app = workflow.compile()
    return app


# Export graph instance
rag_graph = build_rag_graph()


# ---------------------------------------------------------------------------
# 5. Execution / Demonstration
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_question = "What are the major geographic landforms and river systems discussed in the document?"
    print("\n==================== RUNNING LANGGRAPH RAG ====================")
    print(f"Question: {sample_question}\n")

    initial_state: GraphState = {
        "question": sample_question,
        "documents": [],
        "generation": "",
    }

    result = rag_graph.invoke(initial_state)

    print("\n==================== ANSWER ====================")
    print(result["generation"])
    print("\n==================== SOURCES ====================")
    for i, doc in enumerate(result["documents"], start=1):
        page = doc.metadata.get("page", "Unknown")
        snippet = doc.page_content.replace("\n", " ")[:150]
        # Clean non-ascii/unprintable characters for Windows console
        safe_snippet = snippet.encode("ascii", errors="replace").decode("ascii")
        print(f"Source {i} (Page {page}): {safe_snippet}...")
