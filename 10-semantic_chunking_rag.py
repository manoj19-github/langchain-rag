import os
from dotenv import load_dotenv

from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore

# Load environment variables
load_dotenv()

# ==============================================================================
# 1. Embedding Model Setup
# ==============================================================================
# SemanticChunker requires an embedding model to compute cosine distance
# between consecutive sentences and identify topic boundaries.

if os.getenv("OPENAI_API_KEY"):
    from langchain_openai import OpenAIEmbeddings
    print("Using OpenAIEmbeddings (text-embedding-3-small)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
else:
    # 100% Free local embeddings (already downloaded in your project)
    from langchain_huggingface import HuggingFaceEmbeddings
    print("Using HuggingFaceEmbeddings (BAAI/bge-small-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )


# ==============================================================================
# 2. Semantic Chunker Configuration
# ==============================================================================
# Breakpoint threshold strategies:
#   - "percentile": Splits where distance is in the top X percentile (e.g., 95).
#   - "standard_deviation": Splits when distance is > mean + X * std_dev (e.g., 3).
#   - "interquartile": Splits using IQR (interquartile range distance).
#   - "gradient": Splits based on gradient change in distance curve.

text_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # "percentile", "standard_deviation", or "interquartile"
    breakpoint_threshold_amount=95,          # higher percentile = fewer, larger chunks
)


# ==============================================================================
# 3. Sample Multi-Topic Text (Or load from data/test_files)
# ==============================================================================
sample_file_path = "data/test_files/sample.txt"

if os.path.exists(sample_file_path):
    print(f"Reading text from '{sample_file_path}'...")
    with open(sample_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
else:
    # Fallback multi-topic text demonstrating semantic shift detection
    raw_text = """
    Machine Learning is a subset of Artificial Intelligence.
    It enables systems to automatically learn and improve from experience without being explicitly programmed.
    Deep Learning is a specialized branch of Machine Learning inspired by the human brain.
    It uses multi-layered artificial neural networks to process complex data such as images, audio, and text.

    The Eiffel Tower is a world-famous iron lattice monument located in Paris, France.
    It was designed by Gustave Eiffel and completed in 1889 as the centerpiece for the World's Fair.
    Paris is the capital of France and is globally renowned for its art, gastronomy, and cultural landmarks.

    Python is a popular high-level, interpreted programming language created by Guido van Rossum.
    It is widely used in artificial intelligence, data science, web development, and process automation.
    Its simple syntax and vast ecosystem of libraries make it a favorite choice for developers worldwide.
    """

print("\nSplitting text into semantic chunks...")
chunks = text_splitter.create_documents(
    texts=[raw_text],
    metadatas=[{"source": sample_file_path if os.path.exists(sample_file_path) else "inline_sample"}]
)

print(f"\nCreated {len(chunks)} Semantic Chunk(s):\n" + "=" * 65)
for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- [Chunk {i}] (Length: {len(chunk.page_content)} characters) ---")
    print(chunk.page_content.strip())
print("=" * 65)


# ==============================================================================
# 4. In-Memory Vector Store & Retriever
# ==============================================================================
print("\nIndexing semantic chunks into InMemoryVectorStore...")
vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ==============================================================================
# 5. Hugging Face LLM Setup
# ==============================================================================
# Local Hugging Face Pipeline (runs locally on CPU/GPU without API keys or costs)
from langchain_huggingface import HuggingFacePipeline

print("Loading Hugging Face LLM (Qwen/Qwen2.5-0.5B-Instruct)...")
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

# Alternative: If you have Hugging Face API credits and want cloud endpoint:
# from langchain_huggingface import HuggingFaceEndpoint
# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     huggingfacehub_api_token=os.getenv("HF_TOKEN"),
#     temperature=0.1,
#     max_new_tokens=512,
# )


# ==============================================================================
# 6. RAG Chain with LCEL
# ==============================================================================
template = """You are an assistant for question-answering tasks.
Use the following retrieved semantic context to answer the question.
If you don't know the answer, say that you don't know.
Keep your response concise and accurate.

Context:
{context}

Question:
{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ==============================================================================
# 7. Interactive Q&A Loop
# ==============================================================================
print("\n" + "=" * 65)
print(" Semantic Chunking RAG System is Ready!")
print(" Ask questions based on the semantically chunked text.")
print(" Type 'exit' or 'quit' to quit.")
print("=" * 65 + "\n")

while True:
    user_query = input("Enter your question: ").strip()
    if not user_query or user_query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    answer = rag_chain.invoke(user_query)
    print(f"\nAnswer:\n{answer}\n")
    print("-" * 65)
