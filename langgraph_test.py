import os
import weave
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import json
from typing import Literal

memory = MemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(llm_with_tools: ChatOpenAI):
    def inner_chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    return inner_chatbot


def route_tools(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    else:
        return END


def stream_graph_updates(graph, user_input: str, config: dict):
    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()


@weave.op()
def main():
    graph_builder = StateGraph(State)

    tool = TavilySearchResults(max_results=2)
    tools = [tool]
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm_with_tools = llm.bind_tools(tools)
    graph_builder.add_node("chatbot", chatbot(llm_with_tools))

    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    graph = graph_builder.compile(checkpointer=memory, interrupt_before=["tools"])
    config = {"configurable": {"thread_id": "1"}}

    debug_graph_viz = True
    if debug_graph_viz:
        with open("graph.png", "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Exiting...")
            PRINT_SNAPSHOT = False
            if PRINT_SNAPSHOT:
                snapshot = graph.get_state(config)
                print(snapshot)
            stream_graph_updates(graph, user_input, config)
        except Exception as e:
            print(f"Error: {e}")
            user_input = "What do you know about LangGraph?"
            print(f"User: {user_input}")
            break


if __name__ == "__main__":
    weave.init("langgraph_test")
    main()
