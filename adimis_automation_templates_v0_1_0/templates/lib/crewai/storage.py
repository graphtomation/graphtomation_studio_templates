from uuid import uuid4
from pydantic import BaseModel
from asgiref.sync import async_to_sync
from langgraph.store.base import NamespacePath
from langgraph.store.base import Item, SearchItem
from typing import Optional, List, Union, Dict, Any, Literal
from crewai.memory.storage.base_rag_storage import BaseRAGStorage
from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage

from ..storage import PostgresStorage
from ..langgraph.types import StoreConfig


class GraphtomationStorageConfig(BaseModel):
    allow_reset: Optional[bool] = True
    namespace: Optional[NamespacePath] = None
    namespace_prefix: Optional[NamespacePath] = None
    index: Optional[Union[Literal[False], list[str]]] = ["*"]


class GraphtomationStore(PostgresStorage):
    def __init__(
        self, config: StoreConfig, namespace_prefix: Optional[NamespacePath] = None
    ):
        self.namespace_prefix = namespace_prefix
        super().__init__(config)

    def _get_namespace_prefix(self):
        return self.namespace_prefix or ()


class GraphtomationStorage(BaseKnowledgeStorage):
    def __init__(
        self,
        storage: StoreConfig,
        config: Optional[GraphtomationStorageConfig] = None,
    ):
        super().__init__()
        self.config = storage
        self.storage_config = config or GraphtomationStorageConfig()
        self.namespace = self.storage_config.namespace or ()
        self.allow_reset = self.storage_config.allow_reset or True
        self.namespace_prefix = self.storage_config.namespace_prefix or ()
        self.index = self.storage_config.index

        self._store: Optional[GraphtomationStore] = None

    def _get_store(self):
        if not self._store:
            self._store = GraphtomationStore(
                config=self.config, namespace_prefix=self.namespace_prefix
            )
        return self._store

    def search(
        self,
        query,
        limit=3,
        filter=None,
        score_threshold: Optional[Union[int, float]] = 0.35,
    ):
        store = self._get_store()
        items = async_to_sync(store.search)(
            namespace=self.namespace,
            filter=filter,
            limit=limit,
            query=query,
            score_threshold=score_threshold,
        )
        return self._serialize_item(items)

    def save(
        self,
        documents: List[str],
        metadata: Union[Dict[str, Any], List[Dict[str, Any]]],
    ):
        store = self._get_store()

        if not isinstance(documents, list):
            raise ValueError("documents must be a list of strings.")

        if isinstance(metadata, dict):
            metadata_list = [metadata] * len(documents)
        elif isinstance(metadata, list):
            if len(metadata) != len(documents):
                raise ValueError(
                    "Length of metadata list must match the length of documents list."
                )
            metadata_list = metadata
        else:
            raise ValueError("metadata must be a dictionary or a list of dictionaries.")

        formatted_documents = [
            {
                "key": str(uuid4().hex),
                "value": {
                    "content": doc,
                    "metadata": meta,
                },
                "namespace": self.namespace,
                "index": self.index,
            }
            for doc, meta in zip(documents, metadata_list)
        ]

        async_to_sync(store.bulk_create_or_update)(documents=formatted_documents)

    def reset(self):
        if self.allow_reset:
            store = self._get_store()
            async_to_sync(store.reset)(namespace=self.namespace)

    def _serialize_item(self, items: List[Union[Item, SearchItem]]) -> List[Dict]:
        return [item.dict() for item in items]


class GraphtomationRAGStorage(BaseRAGStorage):
    def __init__(
        self,
        type,
        storage: StoreConfig,
        embedder_config=None,
        crew=None,
        config: Optional[GraphtomationStorageConfig] = None,
    ):
        super().__init__(type, config.allow_reset, embedder_config, crew)
        agents = crew.agents if crew else []
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        self.agents = agents
        self.type = type
        self.allow_reset = config.allow_reset

        self.config = storage
        self.namespace = tuple(dict.fromkeys((config.namespace or ()) + ("rag",)))
        self.namespace_prefix = tuple(
            dict.fromkeys(
                (config.namespace_prefix or ()) + ("multi_agent_team_storage",)
            )
        )
        self.index = config.index
        self._initialize_app()

    def _sanitize_role(self, role: str) -> str:
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def save(self, value, metadata):
        self._initialize_app()
        self.storage.save(documents=[value], metadata=metadata)

    def search(self, query, limit=3, filter=None, score_threshold=0.35):
        self._initialize_app()
        return self.storage.search(
            query=[query], limit=limit, filter=filter, score_threshold=score_threshold
        )

    def reset(self):
        self._initialize_app()
        if self.allow_reset:
            self.storage.reset()

    def _initialize_app(self):
        if getattr(self, "storage", None) is None:
            self.storage = GraphtomationStorage(
                storage_config=GraphtomationStorageConfig(
                    config=self.config,
                    namespace_prefix=self.namespace_prefix,
                    namespace=self.namespace,
                    index=self.index,
                )
            )


class GraphtomationLTMStorage:
    def __init__(
        self,
        storage: StoreConfig,
        config: Optional[GraphtomationStorageConfig] = None,
    ):
        self.config = storage
        self.namespace = tuple(dict.fromkeys((config.namespace or ()) + ("ltm",)))
        self.allow_reset = config.allow_reset or True
        self.namespace_prefix = tuple(
            dict.fromkeys(
                (config.namespace_prefix or ()) + ("multi_agent_team_storage",)
            )
        )
        self.index = config.index
        self._initialize_app()

    def _initialize_app(self):
        if getattr(self, "storage", None) is None:
            self.storage = GraphtomationStorage(
                storage_config=GraphtomationStorageConfig(
                    config=self.config,
                    namespace_prefix=self.namespace_prefix,
                    namespace=self.namespace,
                    index=self.index,
                )
            )

    def save(
        self,
        task_description: str,
        metadata: Dict[str, Any],
        datetime: str,
        score: Union[int, float],
    ) -> None:
        self._initialize_db()
        self.storage.save(
            documents=[task_description],
            metadata={
                "metadata": metadata,
                "datetime": datetime,
                "score": score,
            },
        )

    def load(
        self, task_description: str, latest_n: int
    ) -> Optional[List[Dict[str, Any]]]:
        items = self.storage.search(
            limit=latest_n, query=task_description, score_threshold=None
        )
        return [item.get("value", {}).get("metadata", {}) for item in items]

    def reset(self) -> None:
        if self.allow_reset:
            self._initialize_db()
            self.storage.reset()
