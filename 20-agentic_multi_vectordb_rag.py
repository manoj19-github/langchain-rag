import os
import sys
import warnings
from typing import List, Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# 0. Environment & Configuration
# ---------------------------------------------------------------------------
load_dotenv()

# Handle Hugging Face token safety
if os.getenv("HF_TOKEN") == "santraai" or (os.getenv("HF_TOKEN") and not os.getenv("HF_TOKEN").startswith("hf_")):
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

CHROMA_PERSIST_DIR = "./chroma_db_geography_agentic"
FAISS_PERSIST_DIR = "./faiss_index_ai"

# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu", "token": False}
)

# LLM Configuration (using local Ollama with fallback)
llm = ChatOllama(
    model="qwen3:4b-instruct",
    temperature=0.1,
)

# ---------------------------------------------------------------------------
# 1. Multi-Vector Database Setup (3 Distinct Vector DBs)
# ---------------------------------------------------------------------------

import json
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

# ---------------------------------------------------------------------------
# 1. Multi-Vector Database Setup (Ingesting Real Project Data from data/)
# ---------------------------------------------------------------------------

def initialize_faiss_ai_store() -> FAISS:
    """
    Vector DB #1: FAISS (AI & Machine Learning)
    Ingests real project files:
    - data/pdf/attention.pdf (Transformer & Attention Paper)
    - data/text_files/machine_learning.txt (ML Fundamentals)
    - data/text_files/python_intro.txt (Python Basics)
    """
    ai_documents: List[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    # 1. Ingest attention.pdf
    pdf_path = os.path.join("data", "pdf", "attention.pdf")
    if os.path.exists(pdf_path):
        try:
            loader = PyMuPDFLoader(pdf_path)
            raw_docs = []
            for doc in loader.lazy_load():
                raw_docs.append(doc)
                if len(raw_docs) >= 12:  # Load key pages
                    break
            chunks = text_splitter.split_documents(raw_docs)
            for c in chunks:
                c.metadata["domain"] = "AI / Deep Learning"
            ai_documents.extend(chunks)
            print(f"[FAISS Setup] Ingested {len(chunks)} chunks from '{pdf_path}'", flush=True)
        except Exception as e:
            print(f"[FAISS Setup] Error loading {pdf_path}: {e}", flush=True)

    # 2. Ingest machine_learning.txt
    ml_txt_path = os.path.join("data", "text_files", "machine_learning.txt")
    if os.path.exists(ml_txt_path):
        try:
            loader = TextLoader(ml_txt_path, encoding="utf-8")
            ml_docs = loader.load()
            for doc in ml_docs:
                doc.metadata["domain"] = "Machine Learning Basics"
            ai_documents.extend(ml_docs)
            print(f"[FAISS Setup] Ingested '{ml_txt_path}'", flush=True)
        except Exception as e:
            print(f"[FAISS Setup] Error loading {ml_txt_path}: {e}", flush=True)

    # 3. Ingest python_intro.txt
    py_txt_path = os.path.join("data", "text_files", "python_intro.txt")
    if os.path.exists(py_txt_path):
        try:
            loader = TextLoader(py_txt_path, encoding="utf-8")
            py_docs = loader.load()
            for doc in py_docs:
                doc.metadata["domain"] = "Python Programming"
            ai_documents.extend(py_docs)
            print(f"[FAISS Setup] Ingested '{py_txt_path}'", flush=True)
        except Exception as e:
            print(f"[FAISS Setup] Error loading {py_txt_path}: {e}", flush=True)

    if not ai_documents:
        ai_documents.append(Document(
            page_content="Transformers use scaled dot-product attention: Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V.",
            metadata={"source": "Fallback AI Knowledge", "domain": "AI / Deep Learning"}
        ))

    print(f"[VectorStore 1: FAISS] Indexing {len(ai_documents)} documents into FAISS...", flush=True)
    vector_store = FAISS.from_documents(ai_documents, embeddings)
    return vector_store


def initialize_chroma_geography_store() -> Chroma:
    """
    Vector DB #2: ChromaDB (Geography & Earth Sciences)
    Ingests real project files:
    - data/pdf/ccab2-geography.pdf (NCERT Physical Geography)
    - Persists to disk in ./chroma_db_geography_agentic for fast loading
    """
    # Check if existing persistent ChromaDB is populated
    if os.path.exists(CHROMA_PERSIST_DIR) and len(os.listdir(CHROMA_PERSIST_DIR)) > 0:
        print(f"[VectorStore 2: ChromaDB] Loading existing persistent index from '{CHROMA_PERSIST_DIR}'...", flush=True)
        vector_store = Chroma(
            collection_name="geography_agentic_rag",
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        return vector_store

    geo_documents: List[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    pdf_path = os.path.join("data", "pdf", "ccab2-geography.pdf")
    if os.path.exists(pdf_path):
        try:
            print(f"[ChromaDB Setup] Ingesting geography textbook: '{pdf_path}'...", flush=True)
            loader = PyMuPDFLoader(pdf_path)
            raw_docs = []
            for doc in loader.lazy_load():
                raw_docs.append(doc)
                if len(raw_docs) >= 15:  # Ingest top 15 pages (Physiography, drainage, landforms)
                    break
            chunks = text_splitter.split_documents(raw_docs)
            for c in chunks:
                c.metadata["domain"] = "Physical Geography & Earth Sciences"
            geo_documents.extend(chunks)
            print(f"[ChromaDB Setup] Ingested {len(chunks)} chunks from '{pdf_path}'", flush=True)
        except Exception as e:
            print(f"[ChromaDB Setup] Error loading {pdf_path}: {e}", flush=True)

    if not geo_documents:
        geo_documents.append(Document(
            page_content="River drainage basins collect water into main channels forming dendritic and trellis patterns. Alluvial fans form at mountain bases.",
            metadata={"source": "Fallback Geography", "domain": "Physical Geography"}
        ))

    print(f"[VectorStore 2: ChromaDB] Indexing {len(geo_documents)} documents into ChromaDB...", flush=True)
    vector_store = Chroma.from_documents(
        documents=geo_documents,
        embedding=embeddings,
        collection_name="geography_agentic_rag",
        persist_directory=CHROMA_PERSIST_DIR
    )
    return vector_store


def initialize_inmemory_finance_store() -> InMemoryVectorStore:
    """
    Vector DB #3: InMemoryVectorStore (Financial Reports, Corporate & Employee Data)
    Ingests real project files:
    - data/pdf/dummy_company_annual_report_2025_26 (1).pdf (NovaTech Annual Report)
    - data/json_files/company_data.json (TechCorp Departments, Budgets, Projects)
    - data/json_files/employees.json (Employee Directory, Roles, Skills, Salaries)
    """
    finance_documents: List[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    # 1. Ingest Annual Report PDF
    pdf_path = os.path.join("data", "pdf", "dummy_company_annual_report_2025_26 (1).pdf")
    if os.path.exists(pdf_path):
        try:
            loader = PyMuPDFLoader(pdf_path)
            raw_docs = []
            for doc in loader.lazy_load():
                raw_docs.append(doc)
            chunks = text_splitter.split_documents(raw_docs)
            for c in chunks:
                c.metadata["domain"] = "Corporate Financial Report"
            finance_documents.extend(chunks)
            print(f"[InMemory Setup] Ingested {len(chunks)} chunks from '{pdf_path}'", flush=True)
        except Exception as e:
            print(f"[InMemory Setup] Error loading {pdf_path}: {e}", flush=True)

    # 2. Ingest company_data.json
    json_path = os.path.join("data", "json_files", "company_data.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)

            company_name = c_data.get("company", "TechCorp")
            # Ingest departments
            depts = c_data.get("departments", {})
            for dept_name, dept_info in depts.items():
                content = (
                    f"Company: {company_name} | Department: {dept_name.title()}\n"
                    f"Department Head: {dept_info.get('head')}\n"
                    f"Annual Operating Budget: ${dept_info.get('budget'):,}\n"
                    f"Team Size: {dept_info.get('team_size')} employees"
                )
                finance_documents.append(Document(
                    page_content=content,
                    metadata={"source": "data/json_files/company_data.json", "domain": "Corporate Structure & Budgets", "dept": dept_name}
                ))

            # Ingest key employee projects from company_data.json
            for emp in c_data.get("employees", []):
                projects_str = ", ".join([f"{p.get('name')} ({p.get('status')})" for p in emp.get("projects", [])])
                content = (
                    f"Company: {company_name} | Employee: {emp.get('name')} (ID: {emp.get('id')})\n"
                    f"Role: {emp.get('role')}\n"
                    f"Technical Skills: {', '.join(emp.get('skills', []))}\n"
                    f"Assigned Projects: {projects_str}"
                )
                finance_documents.append(Document(
                    page_content=content,
                    metadata={"source": "data/json_files/company_data.json", "domain": "Corporate Staff & Projects", "name": emp.get("name")}
                ))
            print(f"[InMemory Setup] Ingested company departments & projects from '{json_path}'", flush=True)
        except Exception as e:
            print(f"[InMemory Setup] Error parsing {json_path}: {e}", flush=True)

    # 3. Ingest employees.json
    emp_json_path = os.path.join("data", "json_files", "employees.json")
    if os.path.exists(emp_json_path):
        try:
            with open(emp_json_path, "r", encoding="utf-8") as f:
                emp_data = json.load(f)

            employees_list = emp_data.get("employees", [])
            for emp in employees_list:
                content = (
                    f"Employee Profile: {emp.get('name')} (Code: {emp.get('employee_id')})\n"
                    f"Department: {emp.get('department')} | Designation: {emp.get('designation')}\n"
                    f"Monthly Salary: INR {emp.get('salary'):,} | Experience: {emp.get('experience')} years\n"
                    f"Location: {emp.get('city')} | Joining Date: {emp.get('joining_date')}\n"
                    f"Email: {emp.get('email')} | Skills: {', '.join(emp.get('skills', []))}"
                )
                finance_documents.append(Document(
                    page_content=content,
                    metadata={"source": "data/json_files/employees.json", "domain": "HR & Employee Directory", "employee_id": emp.get("employee_id")}
                ))
            print(f"[InMemory Setup] Ingested {len(employees_list)} employee records from '{emp_json_path}'", flush=True)
        except Exception as e:
            print(f"[InMemory Setup] Error parsing {emp_json_path}: {e}", flush=True)

    print(f"[VectorStore 3: InMemory] Indexing {len(finance_documents)} documents into InMemoryVectorStore...", flush=True)
    vector_store = InMemoryVectorStore.from_documents(finance_documents, embeddings)
    return vector_store


# Initialize instances of the 3 Vector Stores and their Retrievers
print("\n=== Initializing Multi-Vector Database Knowledge Bases from Real Data Files ===", flush=True)
faiss_store = initialize_faiss_ai_store()
chroma_store = initialize_chroma_geography_store()
inmemory_store = initialize_inmemory_finance_store()

retriever_faiss = faiss_store.as_retriever(search_kwargs={"k": 3})
retriever_chroma = chroma_store.as_retriever(search_kwargs={"k": 3})
retriever_inmemory = inmemory_store.as_retriever(search_kwargs={"k": 3})
print("=== All 3 Vector Databases Successfully Initialized from Project Files! ===\n", flush=True)


# ---------------------------------------------------------------------------
# 2. LangGraph State Definition
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    target_db: str
    selected_db_name: str
    reasoning: str
    documents: List[Document]
    generation: str


# ---------------------------------------------------------------------------
# 3. LangGraph Nodes Definition
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> Dict[str, Any]:
    """
    Agentic Routing Node:
    Analyzes the user question and routes to the appropriate Vector Database:
    - 'faiss_ai': For AI, Deep Learning, Transformer models, Attention Mechanism, Machine Learning, Python programming.
    - 'chroma_geography': For Geography, Rivers, Landforms, Tectonics, Climate, Earth sciences.
    - 'inmemory_finance': For Company budgets, Departments, Employees, HR directory, Revenue, EBITDA, Cash Flow, Financial reports.
    """
    question = state["question"]
    print(f"\n[Node: Router] Analyzing user prompt: \"{question}\"...", flush=True)

    router_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert AI Routing Agent. Your task is to analyze the user query and select the single "
            "most appropriate Vector Database knowledge base.\n\n"
            "Available Vector Databases:\n"
            "1. 'faiss_ai': AI, Machine Learning fundamentals, Transformers, Attention Mechanism, LLMs, Neural Networks, Python.\n"
            "2. 'chroma_geography': Geography, Landforms, Rivers, Drainage basins, Plate Tectonics, Climate, Physical Earth Science.\n"
            "3. 'inmemory_finance': Company information, Department budgets (Engineering, Data Science), Employee Directory, HR, Salaries, Projects, Corporate Financial Reports (Revenue, EBITDA, Cash Flow).\n\n"
            "Return ONLY a JSON-like format on one line with two fields:\n"
            "ROUTE: <faiss_ai | chroma_geography | inmemory_finance>\n"
            "REASON: <concise explanation>"
        ),
        ("human", "User Query: {question}")
    ])

    router_chain = router_prompt | llm | StrOutputParser()
    try:
        raw_response = router_chain.invoke({"question": question})
    except Exception as e:
        print(f"[Node: Router] LLM call notice: {e}. Using fallback heuristic.", flush=True)
        raw_response = ""

    # Parse route decision with fallback heuristic
    target_db = "faiss_ai"
    selected_db_name = "FAISS (AI & Machine Learning)"
    reasoning = "Defaulted based on topic matching."

    cleaned_resp = raw_response.lower()
    if "chroma_geography" in cleaned_resp or any(w in question.lower() for w in ["river", "geography", "plate", "earth", "mountain", "drainage", "landform", "alluvial", "basin", "himalaya"]):
        target_db = "chroma_geography"
        selected_db_name = "ChromaDB (Geography & Earth Sciences)"
        reasoning = "Query contains geographic / physical earth science concepts."
    elif "inmemory_finance" in cleaned_resp or any(w in question.lower() for w in ["revenue", "ebitda", "cash flow", "finance", "margin", "debt", "balance sheet", "fiscal", "dollar", "$", "employee", "salary", "department", "budget", "techcorp", "novatech", "rahul", "priya", "hr"]):
        target_db = "inmemory_finance"
        selected_db_name = "InMemoryVectorStore (Company, Financial & HR Data)"
        reasoning = "Query contains company, financial, departmental or HR directory concepts."
    else:
        target_db = "faiss_ai"
        selected_db_name = "FAISS (AI & Machine Learning)"
        reasoning = "Query matches Artificial Intelligence / Deep Learning topics."

    print(f"[Node: Router] Decision -> Target Vector DB: [{selected_db_name}]", flush=True)
    print(f"[Node: Router] Reasoning: {reasoning}", flush=True)

    return {
        "question": question,
        "target_db": target_db,
        "selected_db_name": selected_db_name,
        "reasoning": reasoning,
    }


def retrieve_faiss_node(state: AgentState) -> Dict[str, Any]:
    """Retrieval Node for FAISS (AI / ML Knowledge Base)."""
    question = state["question"]
    print(f"\n[Node: Retrieve FAISS] Connecting to FAISS Vector DB embeddings...")
    docs = retriever_faiss.invoke(question)
    print(f"[Node: Retrieve FAISS] Retrieved {len(docs)} relevant chunks from FAISS.")
    return {"documents": docs}


def retrieve_chroma_node(state: AgentState) -> Dict[str, Any]:
    """Retrieval Node for ChromaDB (Geography Knowledge Base)."""
    question = state["question"]
    print(f"\n[Node: Retrieve Chroma] Connecting to ChromaDB Vector DB embeddings...")
    docs = retriever_chroma.invoke(question)
    print(f"[Node: Retrieve Chroma] Retrieved {len(docs)} relevant chunks from ChromaDB.")
    return {"documents": docs}


def retrieve_inmemory_node(state: AgentState) -> Dict[str, Any]:
    """Retrieval Node for InMemory (Finance Knowledge Base)."""
    question = state["question"]
    print(f"\n[Node: Retrieve InMemory] Connecting to InMemory Vector DB embeddings...")
    docs = retriever_inmemory.invoke(question)
    print(f"[Node: Retrieve InMemory] Retrieved {len(docs)} relevant chunks from InMemory.")
    return {"documents": docs}


def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Generation Node:
    Takes retrieved context, attributes the active Vector DB, and generates a grounded response with LLM.
    """
    question = state["question"]
    documents = state["documents"]
    db_name = state["selected_db_name"]
    print(f"\n[Node: Generate] Generating final answer using LLM & context from [{db_name}]...")

    # Format context with citations
    context_entries = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown Source")
        topic = doc.metadata.get("topic", "General")
        snippet = doc.page_content.strip()
        context_entries.append(f"[Chunk {i} | Source: {source} | Topic: {topic}]:\n{snippet}")

    context_str = "\n\n".join(context_entries)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert AI Assistant connected to an Agentic RAG system with multiple specialized vector databases.\n"
            "You are currently answering using context retrieved from: {db_name}.\n\n"
            "Instructions:\n"
            "1. Answer the question accurately based strictly on the provided Context.\n"
            "2. Explicitly reference facts from the context.\n"
            "3. If the context does not contain sufficient information, state what is missing clearly.\n\n"
            "Context:\n{context}"
        ),
        ("human", "Question: {question}")
    ])

    rag_chain = prompt | llm | StrOutputParser()
    try:
        generation = rag_chain.invoke({
            "db_name": db_name,
            "context": context_str,
            "question": question
        })
    except Exception as e:
        generation = f"LLM Generation failed with error: {e}"

    return {"generation": generation}


# ---------------------------------------------------------------------------
# 4. Conditional Edge Function
# ---------------------------------------------------------------------------
def route_decision(state: AgentState) -> str:
    """Conditional routing function determining which retrieval node to execute."""
    target_db = state.get("target_db", "faiss_ai")
    if target_db == "chroma_geography":
        return "retrieve_chroma"
    elif target_db == "inmemory_finance":
        return "retrieve_inmemory"
    else:
        return "retrieve_faiss"


# ---------------------------------------------------------------------------
# 5. Build and Compile the LangGraph
# ---------------------------------------------------------------------------
def build_agentic_rag_graph():
    """Constructs and compiles the multi-vector DB LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve_faiss", retrieve_faiss_node)
    workflow.add_node("retrieve_chroma", retrieve_chroma_node)
    workflow.add_node("retrieve_inmemory", retrieve_inmemory_node)
    workflow.add_node("generate", generate_node)

    # Add Start Edge to Router
    workflow.add_edge(START, "router")

    # Add Dynamic Conditional Branching from Router
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve_faiss": "retrieve_faiss",
            "retrieve_chroma": "retrieve_chroma",
            "retrieve_inmemory": "retrieve_inmemory",
        }
    )

    # Route all retrievals directly to generation node
    workflow.add_edge("retrieve_faiss", "generate")
    workflow.add_edge("retrieve_chroma", "generate")
    workflow.add_edge("retrieve_inmemory", "generate")

    # Terminate after generation
    workflow.add_edge("generate", END)

    # Compile the graph
    app = workflow.compile()
    return app


# Instantiate the graph
agentic_rag_app = build_agentic_rag_graph()


# ---------------------------------------------------------------------------
# 6. Demonstration and Interactive Execution
# ---------------------------------------------------------------------------
def run_query(query: str):
    """Executes a single query through the Agentic LangGraph pipeline."""
    print("=" * 75)
    print(f"USER PROMPT: {query}")
    print("=" * 75)

    initial_state: AgentState = {
        "question": query,
        "target_db": "",
        "selected_db_name": "",
        "reasoning": "",
        "documents": [],
        "generation": "",
    }

    result = agentic_rag_app.invoke(initial_state)

    print("\n" + "-" * 30 + " AGENTIC RAG RESULT " + "-" * 30)
    print(f"Connected Vector DB : {result['selected_db_name']}")
    print(f"Routing Reasoning   : {result['reasoning']}")
    print("-" * 75)
    print("ANSWER:")
    print(result["generation"])
    print("-" * 75)
    print("RETRIEVED CONTEXT SOURCES:")
    for i, doc in enumerate(result["documents"], start=1):
        source = doc.metadata.get("source", "Unknown")
        snippet = doc.page_content.replace("\n", " ")[:130]
        safe_snippet = snippet.encode("ascii", errors="replace").decode("ascii")
        print(f"  [{i}] Source: {source} -> {safe_snippet}...")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    print("\n" + "#" * 75)
    print("#  LANGGRAPH AGENTIC RAG: MULTI-VECTOR DATABASE ROUTING & GENERATION   #")
    print("#" * 75)

    test_queries = [
        "How does the Scaled Dot-Product Attention mechanism compute representations in Transformers?",
        "What are alluvial fans and how do river drainage patterns form?",
        "Who heads the engineering department at TechCorp, what is its budget, and what is employee Rahul Sharma's role and salary?",
    ]

    print("\n>>> Running Automated Multi-Domain Test Demonstrations on Real Project Data <<<\n", flush=True)
    for q in test_queries:
        run_query(q)

    print("\n>>> All Automated Demonstrations Completed Successfully! <<<", flush=True)
