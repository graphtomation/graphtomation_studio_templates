from decouple import config
from rest_framework import serializers


class RunnableConfigSerializer(serializers.Serializer):
    tags = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        help_text="Tags for this call and any sub-calls.",
    )
    metadata = serializers.DictField(
        child=serializers.JSONField(),
        required=False,
        help_text="Metadata for this call and any sub-calls. Keys should be strings, values should be JSON-serializable.",
    )
    run_name = serializers.CharField(
        required=False, help_text="Name for the tracer run for this call."
    )
    max_concurrency = serializers.IntegerField(
        required=False,
        allow_null=True,
        help_text="Maximum number of parallel calls to make. If not provided, defaults to ThreadPoolExecutor's default.",
    )
    recursion_limit = serializers.IntegerField(
        required=False,
        default=25,
        help_text="Maximum number of times a call can recurse. Defaults to 25.",
    )
    configurable = serializers.JSONField(
        required=False,
        help_text="Runtime values for attributes previously made configurable.",
    )
    run_id = serializers.UUIDField(
        required=False,
        allow_null=True,
        help_text="Unique identifier for the tracer run for this call.",
    )

    def validate_metadata(self, value):
        if not all(isinstance(k, str) for k in value.keys()):
            raise serializers.ValidationError("All keys in metadata must be strings.")
        return value

    def validate_recursion_limit(self, value):
        if value is not None and value < 0:
            raise serializers.ValidationError("recursion_limit must be non-negative.")
        return value

    def validate_tags(self, value):
        if len(value) != len(set(value)):
            raise serializers.ValidationError("Tags must be unique.")
        return value


class CheckpointMetadataSerializer(serializers.Serializer):
    source = serializers.ChoiceField(
        choices=["input", "loop", "update"],
        required=False,
        help_text="The source of the checkpoint. Can be 'input', 'loop', or 'update'.",
    )
    step = serializers.IntegerField(
        required=False, help_text="The step number of the checkpoint."
    )
    writes = serializers.JSONField(
        required=False,
        help_text="Writes made between the previous checkpoint and this one, mapping from node name to writes emitted by that node.",
    )
    parents = serializers.JSONField(
        required=False,
        help_text="IDs of the parent checkpoints, mapping from checkpoint namespace to checkpoint ID.",
    )

    def validate_step(self, value):
        if value < -1:
            raise serializers.ValidationError("Step must be -1 or greater.")
        return value


class PregelTaskModelSerializer(serializers.Serializer):
    id = serializers.CharField(help_text="Unique identifier for the task.")
    name = serializers.CharField(help_text="Name of the task.")
    path = serializers.ListField(
        child=serializers.JSONField(), help_text="Path of execution for the task."
    )
    error = serializers.JSONField(
        required=False,
        allow_null=True,
        help_text="Error encountered during the task, if any.",
    )
    interrupts = serializers.ListField(
        child=serializers.JSONField(),
        required=False,
        help_text="Interruptions during execution.",
    )
    state = serializers.JSONField(
        required=False, allow_null=True, help_text="State information for the task."
    )
    result = serializers.DictField(
        required=False, allow_null=True, help_text="Result of the task execution."
    )


class BaseMessageSerializer(serializers.Serializer):
    id = serializers.CharField(required=False, allow_null=True)
    content = serializers.JSONField()
    additional_kwargs = serializers.JSONField(default=dict)
    response_metadata = serializers.JSONField(default=dict)
    type = serializers.CharField()
    name = serializers.CharField(required=False, allow_null=True)


class AIMessageSerializer(BaseMessageSerializer):
    example = serializers.BooleanField(default=False)
    tool_calls = serializers.ListField(child=serializers.JSONField(), default=list)
    invalid_tool_calls = serializers.ListField(
        child=serializers.JSONField(), default=list
    )
    usage_metadata = serializers.JSONField(required=False, allow_null=True)
    type = serializers.CharField(default="ai")


class ChatMessageSerializer(BaseMessageSerializer):
    role = serializers.CharField()
    type = serializers.CharField(default="chat")


class FunctionMessageSerializer(BaseMessageSerializer):
    name = serializers.CharField()
    type = serializers.CharField(default="function")


class HumanMessageSerializer(BaseMessageSerializer):
    example = serializers.BooleanField(default=False)
    type = serializers.CharField(default="human")


class RemoveMessageSerializer(BaseMessageSerializer):
    type = serializers.CharField(default="remove")


class SystemMessageSerializer(BaseMessageSerializer):
    type = serializers.CharField(default="system")


class ToolMessageSerializer(BaseMessageSerializer):
    tool_call_id = serializers.CharField()
    artifact = serializers.JSONField(required=False, allow_null=True)
    status = serializers.ChoiceField(choices=["success", "error"], default="success")
    type = serializers.CharField(default="tool")


# TODO: MAKE IT FASTER LIKE A SENIOR DJANGO DEVELOPER
class StateSnapshotSerializer(serializers.Serializer):
    values = serializers.DictField(
        help_text="Current values of channels.",
        child=serializers.ListField(
            child=serializers.JSONField(),
            help_text="List of serialized messages.",
        ),
    )
    next = serializers.ListField(
        child=serializers.CharField(),
        help_text="Name of the node to execute in each task.",
    )
    config = RunnableConfigSerializer(help_text="Config used to fetch this snapshot.")

    metadata = CheckpointMetadataSerializer(
        required=False,
        allow_null=True,
        help_text="Metadata associated with this snapshot.",
    )

    created_at = serializers.DateTimeField(
        required=False, help_text="Timestamp of snapshot creation.", allow_null=True
    )
    parent_config = RunnableConfigSerializer(
        required=False,
        help_text="Config used to fetch the parent snapshot.",
        allow_null=True,
    )
    tasks = serializers.ListField(
        child=PregelTaskModelSerializer(), help_text="Tasks to execute in this step."
    )

    def validate_values(self, value):
        """
        Optimized validation of the 'values' field by grouping messages by type
        and validating in bulk.
        """
        message_serializers = {
            "ai": AIMessageSerializer,
            "chat": ChatMessageSerializer,
            "function": FunctionMessageSerializer,
            "human": HumanMessageSerializer,
            "remove": RemoveMessageSerializer,
            "system": SystemMessageSerializer,
            "tool": ToolMessageSerializer,
        }

        for channel, messages in value.items():
            if not isinstance(messages, list):
                raise serializers.ValidationError(
                    f"Values for channel '{channel}' must be a list."
                )
            grouped_messages = {}
            for message in messages:
                if not isinstance(message, dict):
                    raise serializers.ValidationError(
                        f"Each message in channel '{channel}' must be serialized to a dictionary."
                    )
                message_type = message.get("type")
                if not message_type:
                    raise serializers.ValidationError(
                        f"Each message in channel '{channel}' must include a 'type' field."
                    )
                grouped_messages.setdefault(message_type, []).append(message)
            for mtype, msgs in grouped_messages.items():
                serializer_class = message_serializers.get(mtype)
                if not serializer_class:
                    raise serializers.ValidationError(
                        f"Unsupported message type '{mtype}' in channel '{channel}'."
                    )
                # Validate all messages of the same type in bulk
                serializer = serializer_class(data=msgs, many=True)
                serializer.is_valid(raise_exception=True)
        return value

    def validate_tasks(self, value):
        """
        Validate the 'tasks' field to ensure all tasks conform to expected structure.
        """
        if not value:
            raise serializers.ValidationError("Tasks cannot be empty.")
        for task in value:
            if "id" not in task or "name" not in task:
                raise serializers.ValidationError(
                    "Each task must include 'id' and 'name' fields."
                )
        return value

    def validate_metadata(self, value):
        """
        Validate metadata to ensure consistency between fields.
        """
        if value:
            if "step" in value and value["step"] < -1:
                raise serializers.ValidationError("Step must be -1 or greater.")
            if "source" in value and value["source"] not in ["input", "loop", "update"]:
                raise serializers.ValidationError(
                    "Source must be one of 'input', 'loop', or 'update'."
                )
        return value

    def validate(self, data):
        """
        Custom validation for the entire serializer.
        """
        if "values" not in data or not data["values"]:
            raise serializers.ValidationError("'values' field cannot be empty.")
        if "next" not in data or not data["next"]:
            raise serializers.ValidationError("'next' field cannot be empty.")
        return data


class UpdateStateModelSerializer(serializers.Serializer):
    config = RunnableConfigSerializer(help_text="Configuration for the state update.")
    values = serializers.DictField(
        child=serializers.JSONField(),
        required=False,
        allow_null=True,
        help_text="Optional dictionary of values to update in the state.",
    )
    as_node = serializers.CharField(
        required=False,
        allow_null=True,
        help_text="Optional name of the node as which to perform the update.",
    )

    def validate(self, data):
        """
        Custom validation for the entire serializer.
        """
        if not data.get("values") and not data.get("as_node"):
            raise serializers.ValidationError(
                "At least one of 'values' or 'as_node' must be provided."
            )
        return data


class GetGraphStateSerializer(serializers.Serializer):
    config = RunnableConfigSerializer(
        help_text="Configuration for fetching the graph state."
    )
    subgraphs = serializers.BooleanField(
        default=False, help_text="Indicates whether to include subgraphs in the state."
    )


class GetGraphStateHistorySerializer(serializers.Serializer):
    config = RunnableConfigSerializer(
        required=True,
        help_text="Configuration used for fetching the graph state history.",
    )
    filter = serializers.DictField(
        child=serializers.JSONField(),
        required=False,
        allow_null=True,
        help_text="Optional filter dictionary for the graph state history.",
    )
    before = RunnableConfigSerializer(
        required=False,
        allow_null=True,
        help_text="Optional configuration before which the graph state history should be fetched.",
    )
    limit = serializers.IntegerField(
        required=False,
        allow_null=True,
        help_text="Optional limit on the number of history entries to fetch.",
    )

    def validate_limit(self, value):
        if value is not None and value <= 0:
            raise serializers.ValidationError("Limit must be a positive integer.")
        return value

    def validate(self, data):
        """
        Custom validation for the entire serializer.
        """
        if not data.get("config"):
            raise serializers.ValidationError("The 'config' field is required.")
        return data


class GraphMetadataSerializer(serializers.Serializer):
    tags = serializers.ListField(
        child=serializers.CharField(max_length=100),
        required=False,
        allow_null=True,
        default=None,
        help_text="List of tags associated with the graph.",
    )
    description = serializers.CharField(
        required=False,
        allow_blank=True,
        allow_null=True,
        default=None,
        max_length=500,
        help_text="Description of the graph.",
    )
    is_published = serializers.BooleanField(
        required=False,
        default=False,
        help_text="Indicates whether the graph is published.",
    )
    version = serializers.CharField(
        required=False,
        allow_blank=True,
        allow_null=True,
        default=None,
        max_length=50,
        help_text="Version of the graph.",
    )
    logo = serializers.CharField(
        required=False,
        allow_blank=True,
        allow_null=True,
        default=None,
        max_length=200,
        help_text="URL or path to the logo of the graph.",
    )
    documentation = serializers.CharField(
        required=False,
        allow_blank=True,
        allow_null=True,
        default=None,
        max_length=500,
        help_text="Documentation of the graph.",
    )


class CheckpointerConfigSerializer(serializers.Serializer):
    name = serializers.ChoiceField(
        choices=["postgres", "memory"], required=False, allow_null=True
    )


class PoolConfigSerializer(serializers.Serializer):
    min_size = serializers.IntegerField(required=True)
    max_size = serializers.IntegerField(required=False, allow_null=True)
    kwargs = serializers.DictField(required=False, allow_null=True)


class ANNIndexConfigSerializer(serializers.Serializer):
    kind = serializers.ChoiceField(
        choices=["hnsw", "ivfflat", "flat"], required=False, allow_null=True
    )
    vector_type = serializers.ChoiceField(
        choices=["vector", "halfvec"], required=False, allow_null=True
    )


class PostgresIndexConfigSerializer(serializers.Serializer):
    ann_index_config = ANNIndexConfigSerializer(required=False, allow_null=True)
    distance_type = serializers.ChoiceField(
        choices=["l2", "inner_product", "cosine"], required=True
    )


class StoreConfigSerializer(serializers.Serializer):
    name = serializers.ChoiceField(
        choices=["postgres", "memory"], required=False, allow_null=True
    )
    pool_config = PoolConfigSerializer(required=False, allow_null=True)
    index = PostgresIndexConfigSerializer(required=False, allow_null=True)


class CompileTemplateSchemaSerializer(serializers.Serializer):
    interrupt_before = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_null=True,
        help_text="List of strings or '*' indicating tasks to interrupt before.",
    )
    interrupt_after = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_null=True,
        help_text="List of strings or '*' indicating tasks to interrupt after.",
    )
    debug = serializers.BooleanField(
        default=False, required=False, help_text="Debug mode flag."
    )
    checkpointer = CheckpointerConfigSerializer(
        required=False, allow_null=True, help_text="Checkpointer configuration."
    )
    store = StoreConfigSerializer(
        required=False, allow_null=True, help_text="Store configuration."
    )


class NodeSerializer(serializers.Serializer):
    id = serializers.CharField(
        max_length=100, help_text="Unique identifier for the node."
    )
    name = serializers.CharField(max_length=100, help_text="Name of the node.")
    data = serializers.JSONField(help_text="Additional data associated with the node.")
    metadata = serializers.JSONField(
        required=False, allow_null=True, help_text="Metadata of the node."
    )


class EdgeSerializer(serializers.Serializer):
    source = serializers.CharField(max_length=100, help_text="Source node identifier.")
    target = serializers.CharField(max_length=100, help_text="Target node identifier.")
    data = serializers.JSONField(
        required=False, allow_null=True, help_text="Additional data for the edge."
    )
    conditional = serializers.BooleanField(
        help_text="Indicates if the edge is conditional."
    )


class SchemaSerializer(serializers.Serializer):
    type = serializers.CharField(max_length=50, help_text="Type of the schema.")
    name = serializers.CharField(
        max_length=100, required=False, help_text="Name of the schema."
    )
    fields = serializers.ListField(
        child=serializers.DictField(
            child=serializers.CharField(), help_text="Field definition."
        ),
        help_text="List of fields in the schema.",
    )


class GraphConfigSerializer(serializers.Serializer):
    nodes = serializers.ListField(
        child=NodeSerializer(),
        help_text="List of nodes in the graph.",
    )
    edges = serializers.ListField(
        child=EdgeSerializer(),
        help_text="List of edges connecting nodes in the graph.",
    )


class GraphSchemaSerializer(serializers.Serializer):
    config = GraphConfigSerializer(
        help_text="Template configuration containing nodes and edges.", required=False
    )
    input_schema = SchemaSerializer(
        required=False, allow_null=True, help_text="Input schema definition."
    )
    output_schema = SchemaSerializer(
        required=False, allow_null=True, help_text="Output schema definition."
    )
    config_schema = SchemaSerializer(
        required=False, allow_null=True, help_text="Configuration schema definition."
    )


class TemplateFormFieldSerializer(serializers.Serializer):
    key = serializers.CharField()
    label = serializers.CharField()
    type = serializers.CharField()
    description = serializers.CharField(required=False, allow_null=True)
    placeholder = serializers.CharField(required=False, allow_null=True)
    defaultValue = serializers.JSONField(required=False, allow_null=True)
    options = serializers.ListField(
        child=serializers.DictField(child=serializers.CharField(allow_null=True)),
        required=False,
        allow_null=True,
    )
    className = serializers.CharField(required=False, allow_null=True)
    required = serializers.BooleanField(required=False, allow_null=True)
    needUserInput = serializers.BooleanField(required=False, allow_null=True)
    required_error_message = serializers.CharField(required=False, allow_null=True)


class TemplateSerializer(serializers.Serializer):
    id = serializers.CharField(
        required=True, help_text="Unique identifier for the graph."
    )
    project_slug = serializers.CharField(
        required=False,
        allow_null=True,
        help_text="Slug of the project the graph belongs to.",
    )
    name = serializers.CharField(required=True, help_text="Name of the graph.")
    metadata = GraphMetadataSerializer(
        required=False, allow_null=True, help_text="Metadata about the graph."
    )
    compile_template_args = CompileTemplateSchemaSerializer(
        required=False, allow_null=True, help_text="Arguments for compiling the graph."
    )
    input_form_fields = TemplateFormFieldSerializer(
        required=False, allow_null=True, help_text="Input form fields.", many=True
    )
    config_form_fields = TemplateFormFieldSerializer(
        required=False, allow_null=True, help_text="Config form fields.", many=True
    )
    node_specific_configs = serializers.JSONField(
        required=False, allow_null=True, help_text="Node-specific configurations."
    )
    template_schema = serializers.JSONField(
        required=True, allow_null=False, help_text="Graph schema."
    )
