import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

load_dotenv()

# ==============================================================================
# 1. LLM Setup
# ==============================================================================
# 100% Free local Hugging Face Pipeline (runs on CPU/GPU without API keys or costs)
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 256,
        "temperature": 0.1,
        "do_sample": True,
        "return_full_text": False,
    },
)

# Alternative: If you want to use Ollama, uncomment below:
# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="llama3.2", temperature=0)

# Alternative: If you want to use Groq, uncomment below:
# from langchain_groq import ChatGroq
# llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_TOKEN"))


# ==============================================================================
# 2. Embedding Model
# ==============================================================================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# ==============================================================================
# 3. Document Loading & Splitting
# ==============================================================================
print("Loading documents from 'data/test_files'...")
dir_loader = DirectoryLoader(
    "data/test_files",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
documents = dir_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", ".", " ", ""],
    length_function=len,
)
chunks = text_splitter.split_documents(documents)
print(f"Loaded {len(documents)} document(s) and split into {len(chunks)} chunk(s).")


# ==============================================================================
# 4. In-Memory Vector Store & Retriever
# ==============================================================================
# In-memory vector store lives entirely in RAM (no disk files / no database server)
print("Indexing chunks into InMemoryVectorStore...")
vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
)

retriever = vector_store.as_retriever(search="similarity",search_kwargs={"k": 3})


def format_docs(docs):
    """Format retrieved documents for insertion into prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


# ==============================================================================
# 5. History-Aware Query Reformulation Sub-chain
# ==============================================================================
# Reformulates follow-up questions (e.g. "When was it created?") into
# standalone queries (e.g. "When was Python created?") using chat history
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

history_aware_retriever = (
    contextualize_q_prompt
    | llm
    | StrOutputParser()
    | retriever
    | format_docs
)


# ==============================================================================
# 6. Conversational QA RAG Chain
# ==============================================================================
qa_system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say you don't know. 
Keep the answer concise (three sentences maximum).

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

rag_chain = (
    RunnablePassthrough.assign(context=history_aware_retriever)
    | qa_prompt
    | llm
    | StrOutputParser()
)


# ==============================================================================
# 7. Interactive Multi-Turn Chat Loop
# ==============================================================================
chat_history = []

print("\n" + "=" * 65)
print(" In-Memory Conversational RAG is Ready!")
print(" Type 'exit' or 'quit' to stop.")
print("=" * 65 + "\n")

while True:
    question = input("Enter your question: ").strip()
    if not question or question.lower() in ["exit", "quit"]:
        print("\nExiting chat. Goodbye!")
        break

    result = rag_chain.invoke({
        "question": question,
        "chat_history": chat_history,
    })

    print(f"\nAnswer: {result}\n")
    print("-" * 65)

    # Maintain conversation history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=result))
