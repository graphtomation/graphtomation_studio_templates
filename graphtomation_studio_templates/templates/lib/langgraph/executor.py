import os
import inspect
import logging
from slugify import slugify
from django.conf import settings
from typing import List, Dict, Optional
from collections.abc import AsyncIterable

from langchain_core.load import dumpd
from psycopg_pool import AsyncConnectionPool
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from .types import (
    GraphInvokeInputState,
    GraphStreamInputState,
    GraphBatchInputState,
    GraphBatchAsCompletedInputState,
    GetGraphState,
    GetGraphStateHistory,
    GetSubgraphs,
    GetGraphSchema,
    GetGraphSchemaResponse,
    StateSnapshotModel,
    TemplateSchema,
    UpdateStateModel,
    CompileTemplateSchema,
)

logger = logging.getLogger(__name__)


class SingleGraphExecutor:
    def __init__(self, args: TemplateSchema):
        self.name = args.name
        self.metadata = args.metadata
        self.state_graph = args.state_graph
        self.compile_template_args = args.compile_template_args
        self.input_form_fields = args.input_form_fields
        self.config_form_fields = args.config_form_fields
        self.node_specific_configs = args.node_specific_configs
        self.template_schema = args.template_schema

    @staticmethod
    async def ainvoke(compiled_graph: CompiledStateGraph, input: GraphInvokeInputState):
        return await compiled_graph.ainvoke(
            config=input.config,
            debug=input.debug,
            input=input.input,
            stream_mode=input.stream_mode,
            output_keys=input.output_keys,
            interrupt_before=input.interrupt_before,
            interrupt_after=input.interrupt_after,
        )

    @staticmethod
    async def abatch(compiled_graph: CompiledStateGraph, input: GraphBatchInputState):
        return await compiled_graph.abatch(
            inputs=input.inputs,
            config=input.config,
            return_exceptions=input.return_exceptions,
        )

    @staticmethod
    async def aget_state(compiled_graph: CompiledStateGraph, input: GetGraphState):
        snapshot = await compiled_graph.aget_state(
            config=input.config, subgraphs=input.subgraphs
        )
        return StateSnapshotModel(
            values=snapshot[0],
            next=snapshot[1],
            config=snapshot[2],
            metadata=snapshot[3],
            created_at=snapshot[4],
            parent_config=snapshot[5],
            tasks=snapshot[6],
        )

    @staticmethod
    async def aupdate_state(
        compiled_graph: CompiledStateGraph, input: UpdateStateModel
    ):
        return await compiled_graph.aupdate_state(
            config=input.config, values=input.values, as_node=input.as_node
        )

    @staticmethod
    async def aget_template_schema(
        compiled_graph: CompiledStateGraph, input: GetGraphSchema
    ) -> GetGraphSchemaResponse:
        graph = await compiled_graph.aget_graph(config=input.config, xray=input.xray)

        input_schema = compiled_graph.get_input_schema(config=input.config)
        output_schema = compiled_graph.get_output_schema(config=input.config)
        config_schema = compiled_graph.config_schema()

        return GetGraphSchemaResponse(
            config=graph,
            input_schema=input_schema,
            output_schema=output_schema,
            config_schema=config_schema,
        )

    @staticmethod
    async def astream(compiled_graph: CompiledStateGraph, input: GraphStreamInputState):
        async for result in compiled_graph.astream(
            input=input.input,
            config=input.config,
            stream_mode=input.stream_mode,
            output_keys=input.output_keys,
            interrupt_before=input.interrupt_before,
            interrupt_after=input.interrupt_after,
            debug=input.debug,
            subgraphs=input.subgraphs,
        ):
            yield result

    @staticmethod
    async def abatch_as_completed(
        compiled_graph: CompiledStateGraph, input: GraphBatchAsCompletedInputState
    ):
        async for result in compiled_graph.abatch_as_completed(
            inputs=input.inputs,
            config=input.config,
            return_exceptions=input.return_exceptions,
        ):
            yield result

    @staticmethod
    async def aget_state_history(
        compiled_graph: CompiledStateGraph, input: GetGraphStateHistory
    ):
        async for snapshot in compiled_graph.aget_state_history(
            config=input.config,
            filter=input.filter,
            before=input.before,
            limit=input.limit,
        ):
            snapshot = StateSnapshotModel(
                values=snapshot[0],
                next=snapshot[1],
                config=snapshot[2],
                metadata=snapshot[3],
                created_at=snapshot[4],
                parent_config=snapshot[5],
                tasks=snapshot[6],
            )
            yield snapshot

    @staticmethod
    async def aget_subgraphs(compiled_graph: CompiledStateGraph, input: GetSubgraphs):
        async for subgraph in compiled_graph.aget_subgraphs(
            namespace=input.namespace, recurse=input.recurse
        ):
            yield subgraph

    @classmethod
    async def _initialize_store(cls, compile_template_args):
        if not compile_template_args or not compile_template_args.get("store"):
            return None

        store_config = compile_template_args["store"]
        conn_string = store_config.get("conn_string") or os.getenv("DB_CONN_STRING")

        print("_initialize_store: store_config: ", store_config)
        if store_config.get("name") == "postgres" and conn_string:
            final_pool_config = {
                **store_config.get(
                    "pool_config", {}
                ),  # Spread the existing pool_config first
                "kwargs": {
                    "autocommit": True,
                    "prepare_threshold": None,
                    **store_config.get("pool_config", {}).get(
                        "kwargs", {}
                    ),  # Merge kwargs
                },
            }

            async with AsyncPostgresStore.from_conn_string(
                conn_string=conn_string,
                pipeline=store_config.get("pipeline", False),
                pool_config=final_pool_config,
                index=store_config.get("index"),
            ) as store:
                await store.setup()
                return store

        elif store_config.get("name") == "memory":
            return InMemoryStore()

        else:
            raise ValueError(f"Unsupported store type: {store_config.get('name')}")

    @classmethod
    async def _initialize_checkpointer(cls, compile_template_args):
        if not compile_template_args or not compile_template_args.get("checkpointer"):
            return None, None

        checkpointer_config = compile_template_args["checkpointer"]
        conn_string = checkpointer_config.get("conn_string") or os.getenv(
            "DB_CONN_STRING"
        )

        if checkpointer_config.get("name") == "postgres" and conn_string:
            pool = AsyncConnectionPool(
                conninfo=conn_string,
                min_size=checkpointer_config.get("min_size", 1),
                max_size=checkpointer_config.get("max_size", 20),
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    **{"autocommit": True, "prepare_threshold": None},
                },
            )
            await pool.open()
            async with pool.connection() as conn:
                checkpointer = AsyncPostgresSaver(conn)
                await checkpointer.setup()
                return checkpointer, pool

        elif checkpointer_config.get("name") == "memory":
            return MemorySaver(), None

        else:
            raise ValueError(
                f"Unsupported checkpointer type: {checkpointer_config.get('name')}"
            )

    @classmethod
    def _compile_state_graph(
        cls, state_graph, compile_template_args, store, checkpointer
    ):
        compiled_graph = state_graph.compile(
            interrupt_before=compile_template_args.get("interrupt_before"),
            interrupt_after=compile_template_args.get("interrupt_after"),
            debug=compile_template_args.get("debug", False),
            checkpointer=checkpointer,
            store=store,
        )
        if not compiled_graph:
            raise ValueError("Failed to compile the state graph.")
        return compiled_graph

    @classmethod
    async def _execute_async_function_await(cls, async_function, compiled_graph, input):
        """
        A pure async function that never yields. It awaits if needed,
        or collects an async iterable into a list.
        """
        result = async_function(compiled_graph, input)
        if inspect.isawaitable(result):
            return await result
        elif isinstance(result, AsyncIterable):
            return [item async for item in result]
        else:
            raise TypeError(
                "The provided async_function must return either an awaitable or an async iterable."
            )

    @classmethod
    async def _execute_async_function_yield(cls, async_function, compiled_graph, input):
        """
        An async generator that yields items if the function returns
        an async iterable, or a single value if it returns a coroutine.
        """
        result = async_function(compiled_graph, input)
        if inspect.isawaitable(result):
            yield await result
        elif isinstance(result, AsyncIterable):
            async for item in result:
                yield item
        else:
            raise TypeError(
                "The provided async_function must return either an awaitable or an async iterable."
            )

    @classmethod
    async def compile_and_execute(
        cls,
        async_function,
        input,
        state_graph,
        compile_template_args,
    ):
        store, checkpointer, pool = None, None, None
        try:
            store = await cls._initialize_store(compile_template_args)
            checkpointer, pool = await cls._initialize_checkpointer(
                compile_template_args
            )
            compiled_graph = cls._compile_state_graph(
                state_graph, compile_template_args, store, checkpointer
            )

            result = async_function(compiled_graph, input)
            return await result
        finally:
            if pool:
                await pool.close()

    @classmethod
    async def compile_and_execute_yield(
        cls,
        async_function,
        input,
        state_graph,
        compile_template_args,
    ):
        store, checkpointer, pool = None, None, None
        try:
            store = await cls._initialize_store(compile_template_args)
            checkpointer, pool = await cls._initialize_checkpointer(
                compile_template_args
            )
            compiled_graph = cls._compile_state_graph(
                state_graph, compile_template_args, store, checkpointer
            )

            result = async_function(compiled_graph, input)
            async for item in result:
                yield item
        finally:
            if pool:
                await pool.close()


class GraphExecutor:
    def __init__(self, templates: List[TemplateSchema] = settings.GRAPH_TEMPLATES):
        self.templates = self._validate_and_slugify_graphs(templates)
        self.template_executors: Dict[str, SingleGraphExecutor] = {}

    def _validate_and_slugify_graphs(
        self, templates: List[TemplateSchema]
    ) -> List[TemplateSchema]:
        name_set = set()
        for graph in templates:
            original_name = graph.name
            if not original_name:
                raise ValueError("Each graph must have a name.")

            slugified_name = slugify(original_name)

            if slugified_name in name_set:
                raise ValueError(
                    f"Duplicate graph name detected: '{slugified_name}'. Please ensure unique names."
                )

            graph.name = slugified_name
            name_set.add(slugified_name)

        return templates

    async def ainvoke(
        self,
        name: str,
        input: GraphInvokeInputState,
        compilation_args: Optional[CompileTemplateSchema] = None,
    ):
        slugified_name = slugify(name)
        template_executor = await self.get_executor(slugified_name)
        result = await SingleGraphExecutor.compile_and_execute(
            SingleGraphExecutor.ainvoke,
            input,
            template_executor.state_graph,
            (
                compilation_args
                if compilation_args is not None
                else template_executor.compile_template_args
            ),
        )
        return dumpd(result)

    async def abatch(
        self,
        name: str,
        input: GraphBatchInputState,
        compilation_args: Optional[CompileTemplateSchema] = None,
    ):
        slugified_name = slugify(name)
        template_executor = await self.get_executor(slugified_name)
        res = await SingleGraphExecutor.compile_and_execute(
            SingleGraphExecutor.abatch,
            input,
            template_executor.state_graph,
            (
                compilation_args
                if compilation_args is not None
                else template_executor.compile_template_args
            ),
        )
        return dumpd(res)

    async def abatch_as_completed(
        self,
        name: str,
        input: GraphBatchAsCompletedInputState,
        compilation_args: Optional[CompileTemplateSchema] = None,
    ):
        slugified_name = slugify(name)
        template_executor = await self.get_executor(slugified_name)
        async for result in SingleGraphExecutor.compile_and_execute_yield(
            SingleGraphExecutor.abatch_as_completed,
            input,
            template_executor.state_graph,
            (
                compilation_args
                if compilation_args is not None
                else template_executor.compile_template_args
            ),
        ):
            yield dumpd(result)

    async def astream(
        self,
        name: str,
        input: GraphStreamInputState,
        compilation_args: Optional[CompileTemplateSchema] = None,
    ):
        slugified_name = slugify(name)
        template_executor = await self.get_executor(slugified_name)
        async for result in SingleGraphExecutor.compile_and_execute_yield(
            SingleGraphExecutor.astream,
            input,
            template_executor.state_graph,
            (
                compilation_args
                if compilation_args is not None
                else template_executor.compile_template_args
            ),
        ):
            yield dumpd(result)

    async def aget_state(
        self,
        name: str,
        input: GetGraphState,
        compilation_args: Optional[CompileTemplateSchema] = None,
    ):
        slugified_name = slugify(name)
        template_executor = await self.get_executor(slugified_name)
        result = await SingleGraphExecutor.compile_and_execute(
            SingleGraphExecutor.aget_state,
            input,
            template_executor.state_graph,
            (
                compilation_args
                if compilation_args is not None
                else template_executor.compile_template_args
            ),
        )
        return dumpd(result)

    async def aupdate_state(
        self,
        name: str,
        input: UpdateStateModel,
        compilation_args: Optional[CompileTemplateSchema] = None,
    ):
        slugified_name = slugify(name)
        template_executor = await self.get_executor(slugified_name)
        res = await SingleGraphExecutor.compile_and_execute(
            SingleGraphExecutor.aupdate_state,
            input,
            template_executor.state_graph,
            (
                compilation_args
                if compilation_args is not None
                else template_executor.compile_template_args
            ),
        )
        return dumpd(res)

    async def aget_state_history(
        self,
        name: str,
        input: GetGraphStateHistory,
        compilation_args: Optional[CompileTemplateSchema] = None,
    ):
        slugified_name = slugify(name)
        template_executor = await self.get_executor(slugified_name)
        async for snapshot in SingleGraphExecutor.compile_and_execute_yield(
            SingleGraphExecutor.aget_state_history,
            input,
            template_executor.state_graph,
            (
                compilation_args
                if compilation_args is not None
                else template_executor.compile_template_args
            ),
        ):
            yield dumpd(snapshot)

    async def aget_subgraphs(
        self,
        name: str,
        input: GetSubgraphs,
        compilation_args: Optional[CompileTemplateSchema] = None,
    ):
        slugified_name = slugify(name)
        template_executor = await self.get_executor(slugified_name)
        async for subgraph in SingleGraphExecutor.compile_and_execute_yield(
            SingleGraphExecutor.aget_subgraphs,
            input,
            template_executor.state_graph,
            (
                compilation_args
                if compilation_args is not None
                else template_executor.compile_template_args
            ),
        ):
            yield dumpd(subgraph)

    async def aget_template_schema(
        self,
        name: str,
        input: GetGraphSchema,
        compilation_args: Optional[CompileTemplateSchema] = None,
    ) -> GetGraphSchemaResponse:
        slugified_name = slugify(name)
        template_executor = await self.get_executor(slugified_name)
        res = await SingleGraphExecutor.compile_and_execute(
            SingleGraphExecutor.aget_template_schema,
            input,
            template_executor.state_graph,
            (
                compilation_args
                if compilation_args is not None
                else template_executor.compile_template_args
            ),
        )
        return dumpd(res)

    async def get_executor(
        self, name: str, compilation_args: Optional[CompileTemplateSchema] = None
    ) -> SingleGraphExecutor:
        slugified_name = slugify(name)
        if slugified_name not in self.template_executors:
            template_args = next(
                (g for g in self.templates if g.name == slugified_name),
                None,
            )
            if not template_args:
                raise ValueError(f"Template '{name}' not found.")

            executor = SingleGraphExecutor(template_args)
            self.template_executors[slugified_name] = executor

        return self.template_executors[slugified_name]

    async def alist_graphs(
        self,
        compilation_args: Optional[CompileTemplateSchema] = None,
    ):
        result = []
        for graph in self.templates:
            name = graph.name
            metadata = graph.metadata
            compile_template_args = graph.compile_template_args

            template_executor = await self.get_executor(name)
            schema = await SingleGraphExecutor.compile_and_execute(
                SingleGraphExecutor.aget_template_schema,
                GetGraphSchema(config=None),
                template_executor.state_graph,
                (
                    compilation_args
                    if compilation_args is not None
                    else template_executor.compile_template_args
                ),
            )
            data = {
                "name": name,
                "metadata": metadata,
                "compile_template_args": compile_template_args,
                "input_form_fields": template_executor.input_form_fields,
                "config_form_fields": template_executor.config_form_fields,
                "node_specific_configs": template_executor.node_specific_configs,
                "template_schema": template_executor.template_schema,
            }
            result.append(self._serialize_template_data(data))
        return result

    async def aget_graph(
        self,
        name: str,
        xray: Optional[bool] = True,
        compilation_args: Optional[CompileTemplateSchema] = None,
    ) -> Dict:
        slugified_name = slugify(name)
        template_executor = await self.get_executor(slugified_name)

        # schema = await SingleGraphExecutor.compile_and_execute(
        #     SingleGraphExecutor.aget_template_schema,
        #     GetGraphSchema(config=None, xray=xray),
        #     template_executor.state_graph,
        #     (
        #         compilation_args
        #         if compilation_args is not None
        #         else template_executor.compile_template_args
        #     ),
        # )

        # Use dot notation to access attributes
        metadata = next(
            (g.metadata for g in self.templates if g.name == slugified_name),
            None,
        )
        compile_template_args = next(
            (
                g.compile_template_args
                for g in self.templates
                if g.name == slugified_name
            ),
            None,
        )

        data = {
            "name": slugified_name,
            "metadata": metadata,
            "compile_template_args": compile_template_args,
            "input_form_fields": template_executor.input_form_fields,
            "config_form_fields": template_executor.config_form_fields,
            "node_specific_configs": template_executor.node_specific_configs,
            "template_schema": template_executor.template_schema,
        }

        return self._serialize_template_data(data)

    def _serialize_template_data(self, template_data):
        serialized_nodes = [
            {
                "id": node.id,
                "name": node.name,
                "data": self._serialize_schema_to_json(node.data),
                "metadata": node.metadata,
            }
            for node_id, node in template_data["template_schema"][
                "config"
            ].nodes.items()
        ]

        serialized_edges = [
            {
                "source": edge.source,
                "target": edge.target,
                "data": edge.data,
                "conditional": edge.conditional,
            }
            for edge in template_data["template_schema"]["config"].edges
        ]

        input_schema = self._serialize_schema_to_json(
            template_data["template_schema"].get("input_schema")
        )

        output_schema = self._serialize_schema_to_json(
            template_data["template_schema"].get("output_schema")
        )
        config_schema = self._serialize_schema_to_json(
            template_data["template_schema"].get("config_schema")
        )

        serialized_data = {
            "name": template_data.get("name"),
            "metadata": template_data.get("metadata", {}),
            "template_schema": {
                "config": {
                    "nodes": serialized_nodes,
                    "edges": serialized_edges,
                },
                "input_schema": input_schema,
                "output_schema": output_schema,
                "config_schema": config_schema,
            },
        }

        return serialized_data

    def _serialize_schema_to_json(self, schema):
        if schema is None:
            return None

        if hasattr(schema, "__annotations__"):
            return {
                "type": "object",
                "name": schema.__name__ if hasattr(schema, "__name__") else None,
                "fields": [
                    {"name": key, "type": str(value)}
                    for key, value in schema.__annotations__.items()
                ],
            }
        else:
            return {"type": "unknown", "value": str(schema)}
