from typing import List, Type
from langchain_core.messages import BaseMessage


def transform_to_raw_message(messages: List[Type[BaseMessage]]) -> str:
    lines = []
    for msg in messages:
        role = msg.type
        role = str(role).capitalize()

        if isinstance(msg.content, str):
            content_str = msg.content
        elif isinstance(msg.content, list):
            content_str = " ".join(str(part) for part in msg.content)
        else:
            content_str = str(msg.content)

        lines.append(f"{role}: {content_str}")

    return "\n".join(lines)
