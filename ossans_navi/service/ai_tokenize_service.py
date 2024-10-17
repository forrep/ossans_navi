import math
import threading

import tiktoken


class AiTokenize:
    @staticmethod
    def image_tokens(width: int, height: int) -> int:
        raise NotImplementedError()

    @staticmethod
    def content_tokens(content: str) -> int:
        raise NotImplementedError()

    @staticmethod
    def messages_tokens(messages: list[dict]) -> int:
        raise NotImplementedError()


class AiTokenizeGpt4o(AiTokenize):
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
                content = message.get("content")
                if isinstance(content, str):
                    summed += len(enc.encode(content))
                elif isinstance(content, list):
                    for v in content:
                        if isinstance(v, dict) and v.get("type") == "text" and isinstance(v.get("text"), str):
                            summed += len(enc.encode(v.get("text")))
            return summed
