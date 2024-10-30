import math
import threading
from typing import Protocol

import tiktoken


class AiTokenize(Protocol):
    @staticmethod
    def image_tokens(width: int, height: int) -> int:
        ...

    @staticmethod
    def content_tokens(content: str) -> int:
        ...

    @staticmethod
    def messages_tokens(messages: list[dict]) -> int:
        ...


class AiTokenizeGpt4o:
    _LOCK = threading.Lock()

    @staticmethod
    def image_tokens(width: int, height: int) -> int:
        """画像解析にかかるトークン数を算出する"""
        if width <= 0 or height <= 0:
            return 0
        return math.ceil(width / 512) * math.ceil(height / 512) * 170 + 85

    @staticmethod
    def content_tokens(content: str) -> int:
        with AiTokenizeGpt4o._LOCK:
            return len(tiktoken.encoding_for_model("gpt-4o").encode(content))

    @staticmethod
    def messages_tokens(messages: list[dict]) -> int:
        enc = tiktoken.encoding_for_model("gpt-4o")
        with AiTokenizeGpt4o._LOCK:
            summed = 0
            for message in messages:
                contents = message.get("content")
                if isinstance(contents, str):
                    summed += len(enc.encode(contents))
                elif isinstance(contents, list):
                    for content in contents:
                        if isinstance(content, dict) and content.get("type") == "text" and isinstance((v := content.get("text")), str):
                            summed += len(enc.encode(v))
            return summed
