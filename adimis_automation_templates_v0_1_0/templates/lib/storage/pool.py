from psycopg_pool import ConnectionPool
from langgraph.store.postgres import PostgresStore


class PostgresStoragePool:
    """Manages a reusable PostgreSQL connection pool for store operations."""

    _pool = None
    _store = None

    @classmethod
    async def setup_pool(cls, conn_string: str, pool_config: dict, store_config: dict):
        """Setup a singleton Postgres connection pool."""
        if cls._pool is None:
            pc = pool_config.copy()
            min_size = pc.pop("min_size", 5)
            max_size = pc.pop("max_size", 20)
            kwargs = pc.pop("kwargs", {})
            kwargs.setdefault("autocommit", True)
            kwargs.setdefault("prepare_threshold", None)
            cls._pool = ConnectionPool(
                conn_string,
                min_size=min_size,
                max_size=max_size,
                kwargs=kwargs,
                **pc,
            )
            cls._store = PostgresStore(conn=cls._pool, index=store_config.get("index"))
        return cls._store

    @classmethod
    async def close_pool(cls):
        """Close the Postgres connection pool."""
        if cls._pool:
            cls._pool.close()
            cls._pool = None
            cls._store = None
