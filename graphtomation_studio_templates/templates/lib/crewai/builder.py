from django.apps import apps
from typing_extensions import TypedDict
from decouple import config as env_config
from typing import Optional, List, Literal

from django.shortcuts import get_object_or_404


from langchain.embeddings import init_embeddings

from crewai.llm import LLM
from crewai import Crew, Process, Agent, Task
from crewai.knowledge.knowledge import Knowledge
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory

from ..utils import pydantic_model_generator, PydanticGeneratorConfig
from .knowledge import GraphtomationKnowledgeSource
from .storage import (
    GraphtomationRAGStorage,
    GraphtomationLTMStorage,
    GraphtomationStorageConfig,
    StoreConfig,
)


# SECTION: MultiAgentTeamBuilder


class KnowledgeSourceConfig(TypedDict, total=False):
    cluster_slug: str
    namespace: Optional[str] = None


class LLMConfig(TypedDict, total=False):
    model_name: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class MemoryConfig(KnowledgeSourceConfig):
    pass


class AgentConfig(TypedDict, total=False):
    role: str
    goal: str
    backstory: str
    max_execution_time: Optional[int]
    use_system_prompt: Optional[bool]
    max_iter: Optional[int]
    max_retry_limit: Optional[int]
    multimodal: Optional[bool]
    code_execution_mode: Optional[str]
    respect_context_window: Optional[bool]
    max_rpm: Optional[int]
    max_tokens: Optional[int]
    allow_delegation: Optional[bool]
    tools: Optional[List]
    knowledge_sources_configs: Optional[list[KnowledgeSourceConfig]]
    cache: Optional[bool]
    llm_config: Optional[LLMConfig]
    function_calling_llm_config: Optional[LLMConfig]
    # tools: Optional[List]


class TaskConfig(TypedDict, total=False):
    name: str
    description: str
    agent_role: Optional[str]
    context_names: Optional[List[str]]
    human_input: Optional[bool]
    max_retries: Optional[int]
    expected_output: Optional[str]
    output_json: Optional[List[PydanticGeneratorConfig]]
    # tools: Optional[List]


class MultiAgentTeamBuilderConfig(TypedDict, total=False):
    agents: List[AgentConfig]
    tasks: List[TaskConfig]
    process: Optional[Process]
    memory: Optional[bool]
    cache: Optional[bool]
    max_rpm: Optional[int]
    name: Optional[str]
    planning: Optional[bool]
    manager_agent_config: Optional[AgentConfig]
    entity_memory_config: Optional[MemoryConfig]
    long_term_memory_config: Optional[MemoryConfig]
    short_term_memory_config: Optional[MemoryConfig]
    input_schema: Optional[List[PydanticGeneratorConfig]] = None
    knowledge_sources_configs: Optional[KnowledgeSourceConfig]
    chat_llm_config: Optional[LLMConfig]
    manager_llm_config: Optional[LLMConfig]
    planning_llm_config: Optional[LLMConfig]
    function_calling_llm_config: Optional[LLMConfig]


class MultiAgentTeamBuilder:
    def __init__(self, config: MultiAgentTeamBuilderConfig):
        self.config: MultiAgentTeamBuilderConfig = config

    def _get_store(self, cluster_slug: str):
        StoreCluster = apps.get_model("src.store", "StoreCluster")
        return get_object_or_404(StoreCluster, slug=cluster_slug)

    def _get_agents(self, agents: List[AgentConfig]):
        return [self._create_agent(agent) for agent in agents]

    def _get_tasks(self, tasks: List[TaskConfig]):
        return [self._create_task(task) for task in tasks]

    def _get_store_config(
        self, cluster_slug: Optional[str] = None
    ) -> Optional[StoreConfig]:
        if not cluster_slug:
            return None

        store = self._get_store(cluster_slug=cluster_slug)
        return {
            "name": "postgres",
            "pipeline": store.pipeline,
            "pool_config": store.pool_config,
            "conn_string": store.db_conn_string or env_config("DB_CONN_STRING"),
            "index": {
                "ann_index_config": {
                    "kind": store.index_config_ann_index_kind,
                    "vector_type": store.index_config_vector_type,
                },
                "distance_type": store.index_config_distance_type,
                "dims": store.index_config_dims,
                "fields": store.index_config_fields,
                "embed": init_embeddings(
                    getattr(store, "index_config_embed_model", None)
                    or "openai:text-embedding-3-large",
                ),
                "embed_model": getattr(store, "index_config_embed_model", None)
                or "openai:text-embedding-3-large",
            },
        }

    def _get_namespace_prefix(self, cluster_slug):
        store = self._get_store(cluster_slug=cluster_slug)
        return tuple(f"projects/{store.project.slug}/stores/{store.slug}".split("/"))

    def _parse_namespace(self, namespace_str: Optional[str] = None):
        if namespace_str:
            namespace = tuple(namespace_str.split("/"))
        else:
            namespace = None

        return namespace

    def _get_knowledge_sources(
        self,
        knowledge_sources_configs: Optional[List[KnowledgeSourceConfig]] = None,
    ):
        if not knowledge_sources_configs:
            return None

        knowledge_sources = []
        for source_config in knowledge_sources_configs:
            cluster_slug = source_config.get("cluster_slug")

            if not cluster_slug:
                return None

            knowledge_sources.append(
                GraphtomationKnowledgeSource(
                    config=GraphtomationStorageConfig(
                        namespace=self._parse_namespace(
                            namespace_str=source_config.get("namespace", None)
                        ),
                        namespace_prefix=self._get_namespace_prefix(
                            cluster_slug=cluster_slug,
                        ),
                    ),
                    store=self._get_store_config(cluster_slug=cluster_slug),
                    chunk_size=source_config.get("chunk_size", 4000),
                    chunk_overlap=source_config.get("chunk_overlap", 20),
                )
            )
        return knowledge_sources

    def _get_storage(
        self,
        config: Optional[MemoryConfig] = None,
        type: Literal["rag", "ltm"] = "rag",
        rag_type: Optional[str] = "short_term",
    ):
        if not config:
            return None

        cluster_slug = config.get("cluster_slug")
        if not cluster_slug:
            return None

        store_config = self._get_store_config(cluster_slug=cluster_slug)

        storage_config = GraphtomationStorageConfig(
            namespace=self._parse_namespace(
                namespace_str=config.get("namespace", None)
            ),
            namespace_prefix=self._get_namespace_prefix(
                cluster_slug=cluster_slug,
            ),
        )

        if type == "ltm":
            return GraphtomationLTMStorage(storage=store_config, config=storage_config)

        if type == "rag":
            return GraphtomationRAGStorage(
                type=rag_type, storage=store_config, config=storage_config
            )

        return None

    def _get_knowledge(
        self,
        config: Optional[StoreConfig] = None,
        knowledge_sources_configs: Optional[List[KnowledgeSourceConfig]] = None,
    ):
        if not config or not knowledge_sources_configs:
            return None

        return Knowledge(
            sources=self._get_knowledge_sources(
                knowledge_sources_configs=knowledge_sources_configs
            ),
            storage=self._get_storage(config=config),
        )

    def _filter_none(self, d: dict) -> dict:
        return {k: v for k, v in d.items() if v is not None}

    def _create_llm(self, config: Optional[LLMConfig] = None):
        if not config:
            return None
        return LLM(
            model=config.get("model_name", "gpt-4o"),
            max_tokens=config.get("max_tokens"),
            temperature=config.get("temperature"),
            top_p=config.get("top_p"),
            api_key=env_config("OPENAI_API_KEY"),
        )

    def _create_agent(self, agent_config: AgentConfig):
        agent_props = {
            "role": agent_config.get("role"),
            "goal": agent_config.get("goal"),
            "backstory": agent_config.get("backstory"),
            "max_execution_time": agent_config.get("max_execution_time"),
            "use_system_prompt": agent_config.get("use_system_prompt", True),
            "max_iter": agent_config.get("max_iter", 25),
            "max_retry_limit": agent_config.get("max_retry_limit", 2),
            "multimodal": agent_config.get("multimodal", False),
            "code_execution_mode": agent_config.get("code_execution_mode", "safe"),
            "respect_context_window": agent_config.get("respect_context_window", True),
            "max_rpm": agent_config.get("max_rpm"),
            "max_tokens": agent_config.get("max_tokens"),
            "allow_delegation": agent_config.get("allow_delegation", False),
            "knowledge_sources": self._get_knowledge_sources(
                agent_config.get("knowledge_sources_configs")
            ),
            "cache": agent_config.get("cache", True),
            "llm": self._create_llm(agent_config.get("llm_config")),
            "function_calling_llm_config": self._create_llm(
                agent_config.get("function_calling_llm_config")
            ),
            "knowledge": self._get_knowledge(
                config=self._get_store_config(
                    cluster_slug=agent_config.get("knowledge_storage_config", {}).get(
                        "cluster_slug"
                    )
                ),
                knowledge_sources_configs=self._get_knowledge_sources(
                    agent_config.get("knowledge_sources_configs")
                ),
            ),
        }
        return Agent(**self._filter_none(agent_props))

    def _create_task(self, task_config: TaskConfig) -> Task:
        """
        Create a Task object while generating an output_json model if provided.
        """
        task_props = {
            "name": task_config.get("name"),
            "description": task_config.get("description"),
            "expected_output": task_config.get("expected_output"),
            "human_input": task_config.get("human_input", False),
            "max_retries": task_config.get("max_retries", 3),
        }

        output_json_config = task_config.get("output_json", None)
        if output_json_config is not None:
            # If output_json_config is already a model (not a list), use it as is.
            if isinstance(output_json_config, list):
                task_props["output_json"] = pydantic_model_generator(output_json_config)
            else:
                task_props["output_json"] = output_json_config

        if self.config:
            all_tasks = self.config.get("tasks", [])
            task_contexts = task_config.get("context_names", [])

            if not isinstance(all_tasks, list) or not isinstance(task_contexts, list):
                raise TypeError("Expected 'tasks' and 'context_names' to be lists")

            task_props["context"] = [
                self._create_task(task)
                for task in all_tasks
                if isinstance(task, dict) and task.get("name") in task_contexts
            ]

            all_agents = self.config.get("agents", [])
            agent_role = task_config.get("agent_role")

            if not isinstance(all_agents, list):
                raise TypeError("Expected 'agents' to be a list")

            task_props["agent"] = next(
                (
                    self._create_agent(agent)
                    for agent in all_agents
                    if isinstance(agent, dict) and agent.get("role") == agent_role
                ),
                None,
            )

        return Task(**task_props)

    def build(self):
        crew_config = self.config

        knowledge_sources = self._get_knowledge_sources(
            knowledge_sources_configs=crew_config.get("knowledge_sources_configs"),
        )

        crew_props = self._filter_none(
            {
                "name": crew_config.get("name"),
                "agents": (
                    self._get_agents(agents=crew_config.get("agents", []))
                    if crew_config.get("agents") is not None
                    else None
                ),
                "tasks": (
                    self._get_tasks(tasks=crew_config.get("tasks", []))
                    if crew_config.get("tasks") is not None
                    else None
                ),
                "process": crew_config.get("process"),
                "memory": crew_config.get("memory"),
                "cache": crew_config.get("cache"),
                "max_rpm": crew_config.get("max_rpm"),
                "planning": crew_config.get("planning", False),
                "manager_agent": (
                    self._create_agent(
                        agent_config=crew_config.get("manager_agent_config")
                    )
                    if crew_config.get("manager_agent_config") is not None
                    else None
                ),
                "short_term_memory": (
                    ShortTermMemory(
                        storage=self._get_storage(
                            config=crew_config.get("short_term_memory_config"),
                            type="rag",
                        )
                    )
                    if crew_config.get("memory")
                    and crew_config.get("short_term_memory_config")
                    else None
                ),
                "long_term_memory": (
                    LongTermMemory(
                        storage=self._get_storage(
                            config=crew_config.get("long_term_memory_config"),
                            type="ltm",
                        )
                    )
                    if crew_config.get("memory")
                    and crew_config.get("long_term_memory_config")
                    else None
                ),
                "entity_memory": (
                    EntityMemory(
                        storage=self._get_storage(
                            config=crew_config.get("entity_memory_config"),
                            type="rag",
                        )
                    )
                    if crew_config.get("memory")
                    and crew_config.get("entity_memory_config")
                    else None
                ),
                "knowledge_sources": (
                    knowledge_sources
                    if knowledge_sources is not None and len(knowledge_sources) > 0
                    else None
                ),
                "knowledge": self._get_knowledge(
                    config=crew_config.get("knowledge_config"),
                    knowledge_sources_configs=crew_config.get(
                        "knowledge_sources_configs"
                    ),
                ),
                "manager_llm": (
                    self._create_llm(config=crew_config.get("manager_llm_config"))
                    if crew_config.get("manager_llm_config") is not None
                    else None
                ),
                "function_calling_llm": (
                    self._create_llm(
                        config=crew_config.get("function_calling_llm_config")
                    )
                    if crew_config.get("function_calling_llm_config") is not None
                    else None
                ),
                "planning_llm": (
                    self._create_llm(config=crew_config.get("planning_llm_config"))
                    if crew_config.get("planning_llm_config") is not None
                    else None
                ),
                "chat_llm": (
                    self._create_llm(config=crew_config.get("chat_llm_config"))
                    if crew_config.get("chat_llm_config") is not None
                    else None
                ),
            }
        )

        return Crew(**crew_props)
