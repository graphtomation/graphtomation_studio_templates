from typing import Any, Optional, Dict
from langchain_core.runnables import RunnableConfig


def get_node_specific_config(
    config: RunnableConfig,
    node: str,
    config_key: Optional[str] = None,
    default: Any = None,
) -> Any:
    """
    Retrieve node-specific configuration from a given configuration dictionary.

    This function assumes that the configuration has the following structure:
      {
          "configurable": {
              "node_specific_configs": {
                  <node>: { ... }
              }
          }
      }

    If 'config_key' is provided, the function returns the value associated with that key
    inside the node configuration. The 'config_key' can also be a nested key path (e.g., "team_config.team_name").
    If a specified key is not found, the function returns the provided default value.
    If 'config_key' is not provided, the entire node configuration is returned.

    Args:
        config (RunnableConfig): The main configuration dictionary.
        node (str): The node key to look for (e.g., "team").
        config_key (Optional[str]): An optional key or nested key path within the node configuration.
        default (Any): The default value to return if the specified config key is not found.

    Returns:
        Any: The node-specific configuration value, or the default value if the key is not found.
    """
    node_specific_configs = config.get("configurable", {}).get(
        "node_specific_configs", {}
    )
    node_config = node_specific_configs.get(node, None)

    if node_config is None and default is not None:
        return default

    if config_key:
        keys = config_key.split(".")
        value = node_config
        for key in keys:
            if isinstance(value, dict):
                if key in value:
                    value = value[key]
                else:
                    return default
            else:
                return default
        return value
    return node_config
