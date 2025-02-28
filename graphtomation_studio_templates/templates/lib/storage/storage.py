import os
from uuid import uuid4
from pydantic import BaseModel
from abc import ABC, abstractmethod
from asgiref.sync import sync_to_async
from langgraph.store.base import GetOp, PutOp
from langgraph.store.base import NamespacePath
from langgraph.store.memory import InMemoryStore
from typing import Literal, Optional, Union, List, Type

from ..langgraph.types import StoreConfig
from ..utils import exponential_retry
from ..storage import PostgresStoragePool


class PostgresStorage(ABC):
    def __init__(self, config: StoreConfig):
        self.config = config
        self.store = None

    @abstractmethod
    async def _get_namespace_prefix(self) -> NamespacePath:
        pass

    def _get_store_config(self):
        """Retrieve and return the store configuration."""
        store_config = self.config or {}
        store_config.setdefault("conn_string", os.getenv("DB_CONN_STRING"))
        return store_config

    def _setup_memory_store(self):
        """Initialize an in-memory store."""
        return InMemoryStore()

    async def _setup_postgres_store(self, store_config):
        """Initialize a Postgres store asynchronously using ClusterItemPostgresPool."""
        conn_string = store_config.get("conn_string")
        if not conn_string:
            raise ValueError("Postgres store requires a connection string")

        pool_config = self._get_pool_config(store_config)

        self.store = await PostgresStoragePool.setup_pool(
            conn_string, pool_config, store_config
        )
        return self.store

    def _get_pool_config(self, store_config):
        """Retrieve and return the pool configuration."""
        pool_config = self._ensure_pool_config(store_config.get("pool_config", {}))
        self._configure_pool_settings(pool_config)
        return pool_config

    def _ensure_pool_config(self, pool_config):
        """Ensure pool_config is a dictionary and initialize if necessary."""
        return pool_config if isinstance(pool_config, dict) else {}

    def _configure_pool_settings(self, pool_config):
        """Set default pool settings."""
        pool_config.setdefault("kwargs", {})
        pool_config["kwargs"].update({"autocommit": True, "prepare_threshold": None})
        pool_config["max_size"] = 20
        pool_config["min_size"] = 5

    def _serialize_documents(self, docs: List) -> dict:
        return {
            "documents": [
                {
                    "key": uuid4().hex,
                    "value": {
                        "metadata": doc.get("document").metadata,
                        "content": doc.get("document").page_content,
                    },
                    "namespace": doc.get("namespace"),
                    "index": doc.get("index") or ["*"],
                }
                for doc in docs
            ]
        }

    @exponential_retry()
    async def search(
        self,
        namespace: NamespacePath,
        query: Optional[str] = None,
        filter: Optional[dict] = None,
        limit: int = 10,
        offset: int = 0,
        ordering: Optional[Literal["desc", "asc"]] = None,
        score_threshold: Optional[float] = None,
    ):
        store = await self.setup()
        store_prefix = await self._get_namespace_prefix()
        namespace_prefix = tuple(dict.fromkeys(store_prefix + namespace))

        items = await store.asearch(
            namespace_prefix,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
        )

        if score_threshold is not None:
            items = [
                item
                for item in items
                if hasattr(item, "score")
                and isinstance(item.score, float)
                and item.score >= score_threshold
            ]

        if ordering in {"asc", "desc"}:
            reverse = ordering == "asc"
            items = sorted(
                items,
                key=lambda x: (
                    getattr(x, "updated_at", 0),
                    getattr(x, "created_at", 0),
                ),
                reverse=reverse,
            )

        return items

    @exponential_retry()
    async def list_namespaces(
        self,
        prefix: Optional[NamespacePath] = None,
        suffix: Optional[NamespacePath] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ):
        store = await self.setup()
        store_prefix = await self._get_namespace_prefix()
        if prefix:
            final_prefix = tuple(dict.fromkeys(store_prefix + prefix))
        else:
            final_prefix = tuple(store_prefix)

        res = store.list_namespaces(
            prefix=final_prefix,
            suffix=suffix,
            max_depth=max_depth,
            limit=limit,
            offset=offset,
        )
        return res

    @exponential_retry()
    async def create_or_update(self, item: dict) -> None:
        store = await self.setup()
        store_prefix = await self._get_namespace_prefix()

        key = item.get("key") or uuid4().hex
        value = item.get("value")
        namespace = tuple(item.get("namespace"))
        index = item.get("index", "$")

        if not key or not value or not namespace:
            raise ValueError("Missing required fields: key, value, and namespace.")

        final_namespace = tuple(dict.fromkeys(tuple(store_prefix) + namespace))

        return await store.aput(
            namespace=final_namespace, key=key, value=value, index=index
        )

    @exponential_retry()
    async def bulk_create_or_update(
        self, documents: list[dict]
    ) -> Union[dict, list[dict]]:
        store = await self.setup()
        store_prefix = await self._get_namespace_prefix()

        ops = []

        for item in documents:
            key = item.get("key") or uuid4().hex
            value = item.get("value")
            namespace = tuple(item.get("namespace"))
            index = item.get("index") or ["*"]

            if not value or not namespace:
                raise ValueError("Missing required fields: value, and namespace.")

            final_namespace = tuple(dict.fromkeys(tuple(store_prefix) + namespace))

            print("bulk_create_or_update: ", key, final_namespace, index, value)
            ops.append(
                PutOp(namespace=final_namespace, key=key, value=value, index=index)
            )

        return await sync_to_async(store.batch)(ops)

    @exponential_retry()
    async def retrieve(self, namespace: NamespacePath, key: str):
        store = await self.setup()
        store_prefix = await self._get_namespace_prefix()
        final_namespace = tuple(dict.fromkeys(tuple(store_prefix) + namespace))

        res = await store.aget(final_namespace, key)
        return res

    @exponential_retry()
    async def bulk_retrieve(self, items: list[dict]):
        store = await self.setup()
        store_prefix = await self._get_namespace_prefix()
        ops = []

        for item in items:
            namespace = tuple(item.get("namespace"))
            key = item.get("key")

            if not namespace or not key:
                raise ValueError("Missing required fields: namespace and key.")

            final_namespace = tuple(dict.fromkeys(tuple(store_prefix) + namespace))

            ops.append(GetOp(final_namespace, key))

        return await sync_to_async(store.batch)(ops)

    @exponential_retry()
    async def delete(self, namespace: NamespacePath, key: str):
        store = await self.setup()
        store_prefix = await self._get_namespace_prefix()
        final_namespace = tuple(dict.fromkeys(tuple(store_prefix) + namespace))

        return await store.adelete(namespace=final_namespace, key=key)

    @exponential_retry()
    async def bulk_delete(self, items: list[dict]):
        store = await self.setup()
        store_prefix = await self._get_namespace_prefix()
        ops = []

        for item in items:
            namespace = tuple(item.get("namespace"))
            key = item.get("key")

            if not namespace or not key:
                raise ValueError("Missing required fields: namespace and key.")

            final_namespace = tuple(dict.fromkeys(tuple(store_prefix) + namespace))

            ops.append(PutOp(final_namespace, key, None))

        return await sync_to_async(store.batch)(ops=ops)

    @exponential_retry()
    async def reset(self, namespace: Optional[NamespacePath] = None):
        store = await self.setup()
        store_prefix = await self._get_namespace_prefix()

        if namespace:
            final_namespace = tuple(dict.fromkeys(tuple(store_prefix) + namespace))
        else:
            final_namespace = tuple(dict.fromkeys(tuple(store_prefix)))

        ops = []
        offset = 0
        limit = 1000

        while True:
            items = await self.search(
                namespace=final_namespace, limit=limit, offset=offset
            )
            if not items:
                break

            for item in items:
                key = item.get("key")
                if not key:
                    continue

                ops.append(PutOp(final_namespace, key, None))

            offset += limit

        return await sync_to_async(store.batch)(ops=ops)

    @exponential_retry()
    async def setup(self):
        """Setup the store based on configuration."""
        if self.store:
            return self.store

        store_config = self._get_store_config()
        store_name = store_config.get("name")

        if store_name == "postgres":
            self.store = await self._setup_postgres_store(store_config)

        elif store_name == "memory":
            self.store = self._setup_memory_store()

        else:
            raise ValueError(f"Unsupported store type: {store_name}")

        return self.store

    @abstractmethod
    async def load_files(
        self,
        args: List[Type[BaseModel]],
    ):
        pass

    @abstractmethod
    async def load_web_urls(self, args: List[Type[BaseModel]]) -> None:
        pass

    @exponential_retry()
    async def serialize_and_insert(self, docs: List[dict]):
        serialized_documents = self._serialize_documents(docs)
        await self.bulk_create_or_update(
            documents=serialized_documents.get("documents")
        )
