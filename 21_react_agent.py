import os
from langchain_core.tools import Tool 
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langgraph.graph import END,StateGraph,START
from typing import Annotated,TypedDict,Sequence
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import ArxivLoader

#1. Setup state 
class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]




from dotenv import load_dotenv
load_dotenv()


# LLM Configuration (using local Ollama with fallback)
llm = ChatOllama(
    model="qwen3:4b-instruct",
    temperature=0.1,
)

#   Create Retriever Tool 

docs = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/").load()
txt_split = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0)
docs = txt_split.split_documents(docs)
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu", "token": False}
)
vector_store = FAISS.from_documents(docs, embedding)
retriever = vector_store.as_retriever()

def retriever_tool_func(query:str)->str:
    print("Using RAG Retriever tool")
    docs = retriever.invoke(query)
    return "\n".join(doc.page_content for doc in docs)

retriever_tool = Tool(
    name="RAGRetriever",
    func = retriever_tool_func,
    description="""
    Use this to get context from documents for the query
    """
)

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(
    top_k_results = 1,
    doc_content_chars_max = 1000,
))

# arxiv = ArxivAPIWrapper(
#     top_k_results=3,
#     doc_content_chars_max=2000
# )

# arxiv_tool = Tool(
#     name="arxiv_search",
#     description="Search arXiv for academic research papers on AI, ML, and computer science.",
#     func=arxiv.run
# )



def arxiv_search(query:str)->str:
    print("Using Arxiv Search tool")
    docs = ArxivLoader(query).load()
    return "\n".join(doc.page_content for doc in docs)

arxiv_tool = Tool(
    name="arxiv_search",
    description="Search arXiv for academic research papers on AI, ML, and computer science.",
    func=arxiv_search
)
    
tools = [retriever_tool,wiki_tool,arxiv_tool]

# Create the native langgraph react agent 


react_node = create_react_agent(llm,tools)

builder = StateGraph(AgentState)
builder.add_node("react_node",react_node)
builder.add_edge(START,"react_node")
builder.add_edge("react_node",END)


app = builder.compile()



if __name__=="__main__":
    user_query="What is agent loop and how does wikipedia decribes autonomous agents?"
    state = {
        "messages":[
            HumanMessage(content=user_query)
        ]
    }
    result = app.invoke(state)
    print("--- Final Answer --- \n",result["messages"][-1].content)
    
    
