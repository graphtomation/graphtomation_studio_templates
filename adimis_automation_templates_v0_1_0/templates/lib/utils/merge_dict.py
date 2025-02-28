from typing import Optional, Dict, Any
import copy


def merge_dict(
    dict1: Optional[Dict[Any, Any]] = None, dict2: Optional[Dict[Any, Any]] = None
) -> Dict[Any, Any]:
    """
    Merges two dictionaries into a single dictionary. In case of conflicts, values from `dict2` take precedence,
    with special handling for merging specific types of values like lists (ensuring uniqueness) and nested dictionaries.

    Args:
        dict1 (Optional[dict]): The first dictionary. Defaults to an empty dictionary if not provided.
        dict2 (Optional[dict]): The second dictionary. Defaults to an empty dictionary if not provided.

    Returns:
        dict: A new dictionary that combines the contents of `dict1` and `dict2`.

    Special Behavior:
    - For keys with lists as values, the lists are merged and deduplicated (with `dict2`'s values appearing first).
    - For keys with dictionaries as values, the dictionaries are recursively merged with `dict2` taking precedence.
    - Other conflicting values are overwritten by `dict2`.
    """
    dict1 = dict1 or {}
    dict2 = dict2 or {}

    def deep_merge(base: Dict[Any, Any], override: Dict[Any, Any]) -> Dict[Any, Any]:
        """Recursively merge `override` into `base`, with `override` winning conflicts."""
        for key, val in override.items():
            if key in base:
                if isinstance(base[key], dict) and isinstance(val, dict):
                    base[key] = deep_merge(base[key], val)
                elif isinstance(base[key], list) and isinstance(val, list):
                    # Merge lists while preserving order and removing duplicates
                    seen = set()
                    base[key] = [
                        x for x in val + base[key] if not (x in seen or seen.add(x))
                    ]
                else:
                    base[key] = copy.deepcopy(val)
            else:
                base[key] = copy.deepcopy(val)
        return base

    return deep_merge(copy.deepcopy(dict2), dict1)
