from uuid import uuid4
from slugify import slugify
from typing_extensions import TypedDict
from langgraph.types import Command, Send
from langgraph.graph.state import RunnableConfig
from langchain_core.runnables.graph import Graph
from langgraph.store.postgres.base import PoolConfig
from pydantic import BaseModel, Field, model_validator
from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.store.postgres.base import PostgresIndexConfig
from typing import (
    Sequence,
    Any,
    Union,
    Optional,
    NotRequired,
    Literal,
    Dict,
    Type,
    Tuple,
    List,
)


class PregelTaskModel(BaseModel):
    id: str
    name: str
    path: Tuple[Union[str, int, Tuple], ...]
    error: Optional[Any] = Field(
        default=None, description="Error encountered during the task."
    )
    interrupts: Tuple[Any, ...] = Field(
        default=(), description="Interruptions during execution."
    )
    state: Optional[Union[None, "StateSnapshotModel", RunnableConfig]] = Field(
        default=None, description="State information."
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Result of the task."
    )

    class Config:
        arbitrary_types_allowed = True


class StateSnapshotModel(TypedDict):
    values: Union[Dict[str, Any], Any] = Field(
        ..., description="Current values of channels."
    )
    next: Tuple[str, ...] = Field(
        ..., description="Name of the node to execute in each task."
    )
    config: RunnableConfig = Field(
        ..., description="Config used to fetch this snapshot."
    )
    metadata: Optional[CheckpointMetadata] = Field(
        default=None, description="Metadata associated with this snapshot."
    )
    created_at: Optional[str] = Field(
        default=None, description="Timestamp of snapshot creation."
    )
    parent_config: Optional[RunnableConfig] = Field(
        default=None, description="Config used to fetch the parent snapshot."
    )
    tasks: Optional[Tuple[Union[PregelTaskModel, Any], ...]] = Field(
        None, description="Tasks to execute in this step."
    )


class GraphInvokeInputState(BaseModel):
    input: Union[Dict[str, Any], Command, Send, Any]
    config: Optional[RunnableConfig] = RunnableConfig(
        configurable={"thread_id": uuid4()}
    )
    stream_mode: Optional[
        Literal["values", "updates", "debug", "messages", "custom"]
    ] = "values"
    output_keys: Optional[Union[str, Sequence[str]]] = None
    interrupt_before: Optional[Union[Literal["*"], Sequence[str]]] = None
    interrupt_after: Optional[Union[Literal["*"], Sequence[str]]] = None
    debug: Optional[bool] = None

    class Config:
        arbitrary_types_allowed = True


class GraphStreamInputState(GraphInvokeInputState):
    subgraphs: bool = False


class GraphBatchInputState(BaseModel):
    inputs: Sequence[Union[Dict[str, Any], Any]]
    config: Optional[Union[RunnableConfig, Sequence[RunnableConfig]]] = None
    return_exceptions: Literal[False] = False

    class Config:
        arbitrary_types_allowed = True


class GraphBatchAsCompletedInputState(GraphBatchInputState):
    pass


class GetGraphState(BaseModel):
    config: RunnableConfig
    subgraphs: bool = False

    class Config:
        arbitrary_types_allowed = True


class GetGraphStateHistory(BaseModel):
    config: RunnableConfig
    filter: Dict[str, Any] | None = None
    before: RunnableConfig | None = None
    limit: int | None = None

    class Config:
        arbitrary_types_allowed = True


class GetSubgraphs(BaseModel):
    namespace: Optional[str] = None
    recurse: bool = False

    class Config:
        arbitrary_types_allowed = True


class GetGraphSchema(BaseModel):
    config: Optional[RunnableConfig] = None
    xray: Optional[Union[int, bool]] = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.config is None:
            self.config = RunnableConfig(configurable={})


class GetGraphSchemaResponse(BaseModel):
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]
    config_schema: Type[BaseModel]
    config: Any

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def validate_components(cls, input_schema, output_schema, config_schema, config):
        if not all(
            isinstance(c, BaseModel)
            for c in [input_schema, output_schema, config_schema]
        ):
            raise TypeError(
                "Input, output, and config schemas must inherit from BaseModel."
            )
        if not isinstance(config, Graph):
            raise TypeError("Graph schema must be an instance of Graph.")


class CheckpointerConfig(TypedDict, total=False):
    name: Optional[Literal["postgres", "memory"]]
    conn_string: NotRequired[Optional[str]]


class IndexConfig(PostgresIndexConfig):
    embed_model: NotRequired[Optional[str]] = None


class StoreConfig(TypedDict, total=False):
    name: Optional[Literal["postgres", "memory"]]
    conn_string: NotRequired[Optional[str]]
    pool_config: NotRequired[Optional[PoolConfig]]
    pipeline: NotRequired[Optional[bool]]
    index: NotRequired[Optional[IndexConfig]]


class CompileTemplateSchema(TypedDict, total=False):
    interrupt_before: Optional[Union[list[str], Literal["*"]]]
    interrupt_after: Optional[Union[list[str], Literal["*"]]]
    debug: NotRequired[Optional[bool]]
    checkpointer: NotRequired[Optional[CheckpointerConfig]]
    store: NotRequired[Optional[StoreConfig]]


class GraphMetadata(BaseModel):
    tags: list[str] = Field(..., description="Tags describing the graph")
    description: str = Field(..., description="Description of the graph")
    is_published: bool = Field(..., description="Publication status of the graph")
    version: str = Field(..., description="Version of the graph")
    logo: str | None = Field(None, description="Optional logo for the graph")
    documentation: Optional[str] = None
    documentation_path: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class TemplateFormField(BaseModel):
    key: str = Field(..., description="Unique key for the form field.")
    label: str = Field(..., description="Label for the form field.")
    type: Union[str, Literal["messages"], Literal["team_builder"]] = Field(
        ..., description="Type of the form field."
    )
    description: Optional[str] = Field(
        None, description="Detailed description of the field."
    )
    placeholder: Optional[str] = Field(None, alias="placeHolder")
    defaultValue: Optional[
        Union[
            str,
            int,
            bool,
            float,
            dict,
            List[Union[str, int, bool, float]],
            Dict[str, Union[str, int, bool, float, List[Union[str, int, bool, float]]]],
        ]
    ] = Field(None, alias="defaultValue")
    options: Optional[List[Dict[str, Union[str, int, bool, Optional[str]]]]] = None
    className: Optional[str] = Field(None, alias="className")
    needUserInput: Optional[bool] = Field(
        False, description="Indicates if the field needs user input."
    )
    required: bool = Field(False, description="Indicates if the field is required.")
    required_error_message: Optional[str] = Field(
        None,
        alias="required_error_message",
        description="Error message if the field is required but missing.",
    )

    class Config:
        arbitrary_types_allowed = True


class NodeInterruptSchema(BaseModel):
    key: str = Field(..., description="Unique key for the value of the interrupt.")
    field: TemplateFormField = Field(..., description="Field for the interrupt.")
    has_question: Optional[bool] = Field(
        False, description="interrupted_data.value has a question attribute."
    )


class NodeData(BaseModel):
    description: str = Field(
        ..., description="Detailed description of the node's function."
    )
    type: Literal["human", "ai", "tool", "start", "end"] = Field(
        ..., description="Type of the node."
    )
    tools: Optional[List[str]] = Field(
        None, description="Tools associated with the node."
    )
    interrupt_schemas: Optional[List[NodeInterruptSchema]] = Field(
        None, description="Schema for handling node interruptions."
    )


class GraphNode(BaseModel):
    label: str = Field(..., description="Human-readable name for the node.")
    id: str = Field(
        default_factory=lambda: "",
        description="Auto-generated slugified identifier for the node.",
    )
    data: NodeData = Field(..., description="Metadata related to the node.")

    def __init__(self, **data):
        super().__init__(**data)
        self.id = self.label

    class Config:
        arbitrary_types_allowed = True


class GraphEdge(BaseModel):
    source: str = Field(..., description="Starting node of the edge.")
    target: str = Field(..., description="Destination node of the edge.")
    conditional: Optional[bool] = Field(
        False, description="Indicates if the transition is conditional."
    )


class GraphSchema(BaseModel):
    nodes: List[GraphNode] = Field(..., description="List of nodes in the graph.")
    edges: List[GraphEdge] = Field(
        ..., description="List of edges defining the graph structure."
    )

    class Config:
        arbitrary_types_allowed = True


class TemplateSchema(BaseModel):
    name: str = Field(..., description="Name of the graph")
    project_slug: Optional[str] = Field(
        None, description="Project slug for the template"
    )
    state_graph: Any = Field(..., description="State graph instance")
    metadata: GraphMetadata = Field(..., description="Metadata of the graph")
    template_schema: GraphSchema = Field(None, description="Schema for the graph")
    compile_template_args: Optional[CompileTemplateSchema] = Field(
        None, description="Compilation arguments for the graph"
    )
    input_form_fields: List[TemplateFormField] = Field(
        ..., description="Schema for the graph input"
    )
    config_form_fields: Optional[List[TemplateFormField]] = Field(
        None, description="State fields for the graph"
    )
    node_specific_configs: Optional[Dict[str, List[TemplateFormField]]] = Field(
        None, description="Node-specific configurations for the graph"
    )

    @model_validator(mode="before")
    def validate_node_specific_configs(cls, values):
        node_specific_configs = values.get("node_specific_configs")
        if node_specific_configs:
            for node_key, form_fields in node_specific_configs.items():
                for field in form_fields:
                    field["needUserInput"] = False
        return values

    class Config:
        arbitrary_types_allowed = True


class UpdateStateModel(BaseModel):
    config: RunnableConfig
    values: Optional[dict] = None
    as_node: Optional[str] = None
