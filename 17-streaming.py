from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage,BaseMessage
from langchain_ollama import ChatOllama



class State(TypedDict):
    messages: Annotated[list, add_messages]


agent = create_agent(
    model=ChatOllama(
        model="qwen3:4b-instruct",
        temperature=0.4
    ),
    tools=[],
    checkpointer=InMemorySaver(),
)


def superbot(state: State):
    return agent.invoke(state)


graph = StateGraph(State)

graph.add_node("bot", superbot)

graph.add_edge(START, "bot")
graph.add_edge("bot", END)

app = graph.compile()


for event in app.stream(
    {
        "messages": [
            HumanMessage(content="Hi my name is Manoj Santra and I like cricket ")
        ]
    },
    config={
        "configurable": {
            "thread_id": "test-1"
        }
    },
    stream_mode="values"
):
    
    print(event["messages"][-1].content)



print("-------------"*5,end="\n")

for event in app.stream(
    {
        "messages": [
            HumanMessage(content="I also like football ")
        ]
    },
    config={
        "configurable": {
            "thread_id": "test-1"
        }
    },
    stream_mode="updates"
):
 print(event["bot"]["messages"][-1].content)