import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage,BaseMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode

load_dotenv()


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "debug-test-1"

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

model = ChatOllama(
    model="qwen3:4b-instruct",
    temperature=0.4
)

def make_default_graph():
    graph_workflow= StateGraph(State)


    def call_model(state):
        return{"messages":[model.invoke(state["messages"])]}
    
    graph_workflow.add_node("bot",call_model)
    graph_workflow.add_edge(START,"bot")
    graph_workflow.add_edge("bot",END)

    return graph_workflow.compile()


agent = make_default_graph()


def make_alternative_graph():
    """Make a tool calling agent""" 
    @tool
    def add_digit(a:float,b:float):
        """ Add two numbers  """
        return a+b
    
      # Actual tools
    tools = [add_digit]
    tool_node = ToolNode(tools)

    # Give tools to the LLM
    model_with_tool = model.bind_tools(tools)

    def bot_with_tool(state):
        return {"messages" : [model_with_tool.invoke(state["messages"])]}
    
    def should_continue(state: State):

            last_message = state["messages"][-1]

            print("\nLAST MESSAGE:")
            print(last_message)

            print("\nTOOL CALLS:")
            print(last_message.tool_calls)

            if last_message.tool_calls:
                return "tools"

            return "end"
    
    graph_workflow = StateGraph(State)
    graph_workflow.add_node("agent",bot_with_tool)
    graph_workflow.add_node("tools",tool_node)
    graph_workflow.add_edge("tools","agent")
    graph_workflow.add_edge(START,"agent")

    # graph_workflow.add_edge("agent",END)

    graph_workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )

   

    return graph_workflow.compile()


# agent_1 = make_default_graph()
agent_2 = make_alternative_graph()

print("-----RUNNING DEFAULT AGENT------\n")
# result = agent.invoke({
#     "messages":[
#         HumanMessage(content="Hi my name is Manoj Santra and I like cricket")
#     ]
# })

# print(result)