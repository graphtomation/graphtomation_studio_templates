from typing import Union, Dict


def replace_secret_keys(json_or_text: Union[Dict, str], keys: Dict[str, str]):
    """
    Replaces occurrences of secret placeholders ($SECRET.<KEY>) with provided values.

    Args:
        json_or_text (dict | str): The input JSON object or string to process.
        keys (dict): A dictionary of keys and their replacement values.

    Returns:
        dict | str: The processed JSON object or string with replaced values.
    """
    import json

    def replace_in_str(input_str):
        for key, value in keys.items():
            input_str = input_str.replace(f"$SECRET.{key}", value)
        return input_str

    if isinstance(json_or_text, str):
        return replace_in_str(json_or_text)
    elif isinstance(json_or_text, dict):
        return json.loads(replace_in_str(json.dumps(json_or_text)))
    else:
        raise ValueError("Input must be a string or a dictionary")
