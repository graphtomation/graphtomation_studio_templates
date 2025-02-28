import os
from typing import Annotated, Optional
from langchain_core.runnables import RunnableConfig

from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]
    template_name: str = "new name"


class Config(TypedDict):
    system_prompt: Optional[str] = None
    model_name: Optional[str] = None


web_search_chatbot = StateGraph(State, config_schema=Config)

tool = DuckDuckGoSearchRun()
tools = [tool]


def chatbot(state: State, config: RunnableConfig):
    """Chatbot that uses web search as tool."""
    chatbot_llm_config = (
        config.get("configurable", {})
        .get("node_specific_configs", {})
        .get("chatbot", {})
        .get("llm")
    )

    if chatbot_llm_config is None:
        raise ValueError("Chatbot llm config is required.")

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=chatbot_llm_config.get("model_name"),
        max_completion_tokens=chatbot_llm_config.get("max_tokens"),
        temperature=chatbot_llm_config.get("temperature"),
        top_p=chatbot_llm_config.get("top_p"),
        system_prompt=chatbot_llm_config.get("system_prompt"),
    ).bind_tools(tools)

    first_message = (
        "system",
        config.get("configurable", {}).get(
            "system_prompt", "You are a helpful assistant."
        ),
    )

    state["messages"].insert(0, first_message)

    return {"messages": [llm.invoke(state["messages"])]}


web_search_chatbot.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
web_search_chatbot.add_node("tools", tool_node)

web_search_chatbot.add_conditional_edges(
    "chatbot",
    tools_condition,
)

web_search_chatbot.add_edge("tools", "chatbot")
web_search_chatbot.add_edge(START, "chatbot")

__all__ = ["web_search_chatbot"]
