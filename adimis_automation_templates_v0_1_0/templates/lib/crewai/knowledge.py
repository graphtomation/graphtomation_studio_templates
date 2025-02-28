from pydantic import Field
from typing import Optional, Any, Optional
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource

from .storage import GraphtomationStorage, GraphtomationStorageConfig, StoreConfig


class GraphtomationKnowledgeSource(BaseKnowledgeSource):
    store: StoreConfig
    config: Optional[GraphtomationStorageConfig] = None
    storage: Optional[GraphtomationStorage] = Field(default=None)

    def model_post_init(self, __context__: Optional[dict[str, Any]] = None):
        if not self.store:
            raise ValueError(
                "GraphtomationKnowledgeSource requires a store configuration."
            )

        if not self.storage:
            self.storage = GraphtomationStorage(storage=self.store, config=self.config)
