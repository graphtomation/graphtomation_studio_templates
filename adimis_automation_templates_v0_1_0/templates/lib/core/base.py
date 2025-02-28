from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Optional, TypeVar, Generic, Annotated, List

T = TypeVar("T")


class BaseGraphConfig(Generic[T], TypedDict):
    node_specific_configs: Optional[T]


class BaseGraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
