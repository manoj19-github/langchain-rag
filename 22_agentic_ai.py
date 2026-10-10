import os
os.environ.setdefault("USER_AGENT", "rag_tuts/1.0")
from langchain_core.tools import Tool, create_retriever_tool
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
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated,TypedDict,Sequence,Literal
from typing_extensions import  TypedDict
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import ArxivLoader
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel,Field


#1. Setup state 
class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]
    relevance: str


langgraph_urls = [
    "https://docs.langchain.com/oss/python/langgraph/overview",
    "https://docs.langchain.com/oss/python/langgraph/quickstart",
    "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "https://docs.langchain.com/oss/python/langgraph/graph-api",
    "https://docs.langchain.com/oss/python/langgraph/use-graph-api",
    "https://docs.langchain.com/oss/python/langgraph/functional-api",
    "https://docs.langchain.com/oss/python/langgraph/persistence",
    "https://docs.langchain.com/oss/python/langgraph/memory",
    "https://docs.langchain.com/oss/python/langgraph/interrupts",
    "https://docs.langchain.com/oss/python/langgraph/human-in-the-loop",
]


docs = [WebBaseLoader(url).load() for url in langgraph_urls]
docs = [d for subdoc in docs for d in subdoc]
txt_split = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0)
docs = txt_split.split_documents(docs)
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu", "token": False}
)
vector_store = FAISS.from_documents(docs, embedding)
retriever = vector_store.as_retriever()

#2. Multi Model Setup
llm = ChatOllama(
    model="qwen3:4b-instruct",
    temperature=0.1,
)

# Helper functions
def retriever_tool_func(query:str)->str:
    print("Using RAG Retriever tool")
    docs = retriever.invoke(query)
    return "\n".join(doc.page_content for doc in docs)

# retriever_tool = Tool(
#     name="RAGRetriever",
#     func = retriever_tool_func,
#     description="""
#     Use this to get context from documents for the query
#     """
# )


#  ---------------------  Retriever to Retrieval Tools -----------------------
retriever_tool = create_retriever_tool(
    retriever,
    name="RAGRetriever",
    description="""
    Use this to get context from documents for the query
    """
)


langchain_urls = [
    "https://docs.langchain.com/oss/python/langchain/overview",
    "https://docs.langchain.com/oss/python/langchain/quickstart",
    "https://docs.langchain.com/oss/python/langchain/agents",
    "https://docs.langchain.com/oss/python/langchain/models",
    "https://docs.langchain.com/oss/python/langchain/tools",
    "https://docs.langchain.com/oss/python/langchain/short-term-memory",
    "https://docs.langchain.com/oss/python/langchain/long-term-memory",
    "https://docs.langchain.com/oss/python/langchain/middleware",
    "https://docs.langchain.com/oss/python/langchain/retrieval",
    "https://docs.langchain.com/oss/python/langchain/rag",
    "https://docs.langchain.com/oss/python/langchain/structured-output",
    "https://docs.langchain.com/oss/python/langchain/human-in-the-loop",
]


langchain_docs = [WebBaseLoader(url).load() for url in langchain_urls]
langchain_docs = [d for subdoc in langchain_docs for d in subdoc]
txt_split = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0)
langchain_docs = txt_split.split_documents(langchain_docs)
langchain_vector_store = FAISS.from_documents(langchain_docs, embedding)
langchain_retriever = langchain_vector_store.as_retriever()

langchain_docs_retriever_tool = create_retriever_tool(
    langchain_retriever,
    name="LangchainDocsRetriever",
    description="""
    Use this to get context from Langchain documents for the query
    """
)

#   Langchain Blogs sperate  Vector DB 


tools = [retriever_tool,langchain_docs_retriever_tool]

#  ------------------ Agentic AI - Using prebuilt Agent ---------------------------

react_agent = create_react_agent(llm,tools)


# Langgraph Workflow 

def agent(state:AgentState):
    messages = state["messages"]
    model_withdevtools = llm.bind_tools(tools)
    response = model_withdevtools.invoke(messages)
    return {"messages":[response]}
    

def grade_documents(state:AgentState):
    """
    Determines whether the retrieved documents are relevant to the question.
    Updates state with relevance ('yes' or 'no').

    Args:
        state (messages) : The current state
    Returns:
        dict: The updated state with relevance score
    """

    class grade(BaseModel):
        """ 
        Binary score for relevance check 
        
        """

        binary_score:str = Field(description="Relevance score 'yes' or 'no'")
    llm = ChatOllama(
        model="qwen3:4b-instruct",
        temperature=0.1,
    )
    prompt = PromptTemplate(
        template="""
        You are a grader assessing relevance of a retrieved document to a user question.\n 
        Here is the retrieved document: \n \n {context} \n \n
        If the document contains relevance keywords or is semantically related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Return only the binary score, no additional text. \n
        """,
        input_variables=["context","question"]
    )

    llm_with_tool = llm.with_structured_output(grade)
    chain = prompt | llm_with_tool
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    scored_result = chain.invoke({"question":question,"context":docs})
    score = scored_result.binary_score.lower()

    if score == "yes":
        print(" --- NODE GRADE: DOCS RELEVANT")
    else:
        print(" --- NODE GRADE: DOCS NOT RELEVANT")
    
    return {"relevance": score}


def decide_to_generate(state:AgentState)->Literal["generate","rewrite"]:
    """
    Conditional edge router based on the relevance score in state.

    Args:
        state: The current state
    Returns:
        str: Decision to either generate or rewrite
    """
    relevance = state.get("relevance", "no")
    if relevance == "yes":
        print(" --- DECISION: GENERATE")
        return "generate"
    else:
        print(" --- DECISION: REWRITE")
        return "rewrite"
    
    
    

def generate(state):
    """ 

    Generate answer

    Args:
        state (messages) : The current state
    Returns:
        str: The updated message

    """

    print(" ---  GENERATE")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    context = last_message.content
    llm = ChatOllama(
        model="qwen3:4b-instruct",
        temperature=0.1,
    )
    prompt = PromptTemplate(
    template="""
    You are a technical assistant specializing in LangChain
    and LangGraph.

    Answer the question using only the supplied documentation.
    If the documentation does not contain the answer,
    clearly state that you do not know.

    Question:
    {question}

    Context:
    {context}

    Answer:
    """,
        input_variables=["question", "context"]
    )

    def format_docs(docs):
        if isinstance(docs, str):
            return docs
        return "\n\n".join(getattr(document, "page_content", str(document)) for document in docs)
    
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({
        "question":question,
        "context":format_docs(context)
        })
    return {"messages":[AIMessage(content=response)]}
    


def rewrite(state:AgentState):
    """
    Transform the query to produce a better question. 

    Args:
        state (messages) : The current state
    Returns:
        dict: The updated state with re-phrased question

    """
    print(" ---- Transformed Query")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f"""\n
            Look at the input and try to reason about the underlying semantic intent / meaning. \n 
            Here is the initial question:
            \n ------- \n 
            {question}

            \n -------- \n 
            Formulate am improved question : 
            
            """
        )

    ]

    llm = ChatOllama(
        model="qwen3:4b-instruct",
        temperature=0.1,
    )

    response = llm.invoke(msg)
    rewritten_query = response.content
    print("ReWritten Query" ,rewritten_query)
    return {"messages":[HumanMessage(content=rewritten_query)]}


#   define a new graph 

workflow = StateGraph(AgentState)

retrieve = ToolNode([retriever_tool,langchain_docs_retriever_tool])
# Add nodes
workflow.add_node("agent",agent)
workflow.add_node("retrieve",retrieve)
workflow.add_node("grade_documents",grade_documents)
workflow.add_node("rewrite",rewrite)
workflow.add_node("generate",generate)

# Add edges with conditions 
workflow.add_edge(START,"agent")


# conditional edge for generate or rewrite
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools":"retrieve",
        END:END
    }
)

# After retrieving, proceed to grade_documents node
workflow.add_edge("retrieve", "grade_documents")

# From grade_documents node, route conditionally
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)
workflow.add_edge("generate",END)
workflow.add_edge("rewrite","agent")


app = workflow.compile()

response = app.invoke({"messages": [HumanMessage(content="How to install Langgraph?")]})
print("\n  ---- Final Answer : \n",response["messages"][-1].content)

    




