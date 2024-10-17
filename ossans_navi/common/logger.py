from typing import Any


def shrink_message(message: str | list | dict | Any, limit: int = 100) -> str | list | dict | Any:
    if limit <= 3:
        raise ValueError("limit must be greater than or equal to 4. limit=" + limit)
    if isinstance(message, dict):
        return {k: shrink_message(v) for (k, v) in message.items()}
    if isinstance(message, list):
        return [shrink_message(v) for v in message]
    if isinstance(message, str):
        return message[:limit - 3] + "..."
    return message
