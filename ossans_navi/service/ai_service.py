import base64
import dataclasses
import itertools
import json
import logging
import threading
import time
from collections import defaultdict
from enum import Enum
from operator import itemgetter
from typing import Any, Iterable, Optional, overload

from google import genai
from google.genai import types
from openai import AzureOpenAI, InternalServerError, OpenAI, RateLimitError
from openai.types.chat import (ChatCompletion, ChatCompletionContentPartParam, ChatCompletionMessageParam, ChatCompletionMessageToolCallParam,
                               ChatCompletionToolParam, completion_create_params)

from ossans_navi import config
from ossans_navi.common.logger import shrink_message
from ossans_navi.config import AiServiceType
from ossans_navi.service.ai_tokenize_service import AiTokenize, AiTokenizeGpt4o
from ossans_navi.type import ossans_navi_types

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


@dataclasses.dataclass
class AiModel:
    name: str
    ai_service_type: AiServiceType
    cost_in: float
    cost_out: float
    tokenizer: AiTokenize
    tokens_in: int = dataclasses.field(default=0, init=False)
    tokens_out: int = dataclasses.field(default=0, init=False)

    def get_total_cost(self) -> float:
        return self.cost_in * self.tokens_in / 1_000_000 + self.cost_out * self.tokens_out / 1_000_000


@dataclasses.dataclass
class AiModels:
    low_cost: AiModel = dataclasses.field(init=False)
    high_quality: AiModel = dataclasses.field(init=False)

    @staticmethod
    def new() -> 'AiModels':
        models = AiModels()
        match config.AI_SERVICE_TYPE:
            case AiServiceType.OPENAI:
                models.low_cost = AiModel(
                    config.OPENAI_MODEL_LOW_COST,
                    config.AI_SERVICE_TYPE,
                    config.OPENAI_MODEL_LOW_COST_IN,
                    config.OPENAI_MODEL_LOW_COST_OUT,
                    AiTokenizeGpt4o
                )
                models.high_quality = AiModel(
                    config.OPENAI_MODEL_HIGH_QUALITY,
                    config.AI_SERVICE_TYPE,
                    config.OPENAI_MODEL_HIGH_QUALITY_IN,
                    config.OPENAI_MODEL_HIGH_QUALITY_OUT,
                    AiTokenizeGpt4o
                )
            case AiServiceType.AZURE_OPENAI:
                models.low_cost = AiModel(
                    config.AZURE_OPENAI_MODEL_LOW_COST,
                    config.AI_SERVICE_TYPE,
                    config.AZURE_OPENAI_MODEL_LOW_COST_IN,
                    config.AZURE_OPENAI_MODEL_LOW_COST_OUT,
                    AiTokenizeGpt4o
                )
                models.high_quality = AiModel(
                    config.AZURE_OPENAI_MODEL_HIGH_QUALITY,
                    config.AI_SERVICE_TYPE,
                    config.AZURE_OPENAI_MODEL_HIGH_QUALITY_IN,
                    config.AZURE_OPENAI_MODEL_HIGH_QUALITY_OUT,
                    AiTokenizeGpt4o
                )
            case AiServiceType.GEMINI:
                models.low_cost = AiModel(
                    config.GEMINI_MODEL_LOW_COST,
                    config.AI_SERVICE_TYPE,
                    config.GEMINI_MODEL_LOW_COST_IN,
                    config.GEMINI_MODEL_LOW_COST_OUT,
                    AiTokenizeGpt4o
                )
                models.high_quality = AiModel(
                    config.GEMINI_MODEL_HIGH_QUALITY,
                    config.AI_SERVICE_TYPE,
                    config.GEMINI_MODEL_HIGH_QUALITY_IN,
                    config.GEMINI_MODEL_HIGH_QUALITY_OUT,
                    AiTokenizeGpt4o
                )
            case _:
                raise NotImplementedError("Unknown Service.")
        return models

    def models(self) -> list[AiModel]:
        return [
            model
            for model in [
                getattr(self, model_name) for model_name in vars(self).keys()
            ]
            if isinstance(model, AiModel)
        ]

    def get_total_cost(self) -> float:
        return sum([model.get_total_cost() for model in self.models()])


class AiPromptRole(Enum):
    ASSISTANT = "assistant"
    USER = "user"


@dataclasses.dataclass
class AiPromptRagInfo:
    contents: list | dict
    words: list[str]


@dataclasses.dataclass
class AiPromptImage:
    data: bytes

    @property
    def image_uri(self) -> str:
        return f"data:image/png;base64,{base64.b64encode(self.data).decode()}"


@dataclasses.dataclass
class AiPromptContent:
    data: str | dict[str, Any]
    images: list[AiPromptImage] = dataclasses.field(default_factory=list)

    @property
    def is_dict(self) -> bool:
        return isinstance(self.data, dict)

    @property
    def text(self) -> str:
        if isinstance(self.data, dict):
            return self.data["content"]
        else:
            return self.data

    @property
    def detail(self) -> dict[str, Any]:
        if isinstance(self.data, dict):
            return {k: v for (k, v) in self.data.items() if k != "content"}
        else:
            return {}


@dataclasses.dataclass
class AiPromptFunctionId:
    call: int = dataclasses.field(default=0, init=False)
    response: int = dataclasses.field(default=0, init=False)

    @property
    def call_id(self) -> int:
        self.call = self.call + 1
        return self.call

    @property
    def response_id(self) -> int:
        self.response = self.response + 1
        return self.response


@dataclasses.dataclass
class AiPromptMessage:
    role: AiPromptRole
    content: AiPromptContent
    name: Optional[str] = dataclasses.field(default=None)

    def to_openai_messages(self, function_id: AiPromptFunctionId, rag_info: Optional[AiPromptRagInfo] = None) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        message: Optional[ChatCompletionMessageParam] = None
        if self.role == AiPromptRole.ASSISTANT:
            message = {
                "role": AiPromptRole.ASSISTANT.value,
                "content": self.content.text,
            }
        elif self.role == AiPromptRole.USER:
            # USERロールの場合のみ画像を入力する
            # OpenAI は ASSISTANTロールでの画像を入力に未対応、誤って入力するとエラーとなる
            # ASSISTANT の応答文に画像リンクが含まれると Slack 上ではメッセージに画像が添付される、よって ASSISTANTロールのメッセージにも画像が付くケースはある
            message_contents: list[ChatCompletionContentPartParam] = []
            message_contents.append(
                {
                    "type": "text",
                    "text": self.content.text,
                }
            )
            for image in self.content.images:
                message_contents.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image.image_uri
                        }
                    }
                )
            message = {
                "role": AiPromptRole.USER.value,
                "content": message_contents,
            }
        else:
            raise ValueError("Unknown AiPromptRole: " + str(self.role))
        if self.name:
            message["name"] = self.name
        messages.append(message)

        if self.content.is_dict or rag_info:
            tool_calls: list[ChatCompletionMessageToolCallParam] = []
            if self.content.is_dict:
                tool_calls.append(
                    {
                        "id": f"call_{function_id.call_id}",
                        "type": "function",
                        "function": {
                            "name": "get_last_message_detail",
                            "arguments": "",
                        },
                    }
                )
            if rag_info:
                tool_calls.append(
                    {
                        "id": f"call_{function_id.call_id}",
                        "type": "function",
                        "function": {
                            "name": "get_related_information",
                            "arguments": json.dumps(rag_info.words, ensure_ascii=False),
                        },
                    }
                )
            messages.append(
                {
                    "role": AiPromptRole.ASSISTANT.value,
                    "content": None,
                    "tool_calls": tool_calls,
                }
            )
        if self.content.is_dict:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{function_id.response_id}",
                    "content": json.dumps(self.content.detail, ensure_ascii=False, separators=(',', ':')),
                },
            )
        if rag_info:
            messages.append(
                {
                    "tool_call_id": f"call_{function_id.response_id}",
                    "role": "tool",
                    "content": json.dumps(
                        rag_info.contents if len(rag_info.contents) > 0 else {
                            "status": "No valid information was found in the get_related_information results, please respond in general terms."
                        },
                        ensure_ascii=False,
                        separators=(',', ':')
                    ),
                },
            )
        return messages

    def to_gemini_content(self, rag_info: Optional[AiPromptRagInfo] = None) -> list[types.ContentDict]:
        contents: list[types.ContentDict] = []
        parts: list[types.PartDict] = []
        parts.append({"text": self.content.text})
        if self.role != AiPromptRole.ASSISTANT:
            # AiPromptRole.ASSISTANT ではない場合のみ画像を読み込む
            # モデルが画像生成機能を持たない現時点では、AiPromptRole.ASSISTANT に画像データが付随するケースは電文を偽造しない限りない
            # そのため現時点では画像を送信しても処理対象にならない、そのため送らない
            parts.extend(
                [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image.data,
                        }
                    }
                    for image in self.content.images
                ]
            )
        contents.append({
            "role": "model" if self.role == AiPromptRole.ASSISTANT else self.role.value,
            "parts": parts,
        })

        if self.content.is_dict or rag_info:
            parts = []
            if self.content.is_dict:
                parts.append(
                    {
                        "function_call": {
                            "name": "get_last_message_detail",
                            "args": {},
                        }
                    }
                )
            if rag_info:
                parts.append(
                    {
                        "function_call": {
                            "name": "get_related_information",
                            "args": {
                                "terms": rag_info.words
                            },
                        }
                    }
                )
            contents.append(
                {
                    "role": "model",
                    "parts": parts,
                }
            )
            parts = []
            if self.content.is_dict:
                parts.append(
                    {
                        "function_response": {
                            "name": "get_last_message_detail",
                            "response": self.content.detail,
                        }
                    }
                )
            if rag_info:
                parts.append(
                    {
                        "function_response": {
                            "name": "get_related_information",
                            "response": {
                                **(
                                    {
                                        "status": (
                                            "No valid information was found in the get_related_information results, "
                                            + "please respond in general terms."
                                        )
                                    } if len(rag_info.contents) == 0 else {}
                                ),
                                **(
                                    {
                                        "contents": rag_info.contents,
                                    } if len(rag_info.contents) > 0 else {}
                                ),
                            }
                        }
                    }
                )
            contents.append(
                {
                    "role": "user",
                    "parts": parts,
                }
            )

        return contents


@dataclasses.dataclass
class AiPrompt:
    system: str
    messages: list[AiPromptMessage]
    schema: Optional[types.Schema] = dataclasses.field(default=None)
    choices: int = dataclasses.field(default=1)
    rag_info: Optional[AiPromptRagInfo] = dataclasses.field(default=None)

    @property
    def is_json(self) -> bool:
        return self.schema is not None

    @overload
    @staticmethod
    def _convert_type_to_lower(values: dict[str, Any]) -> dict[str, Any]:
        ...

    @overload
    @staticmethod
    def _convert_type_to_lower(values: list[Any]) -> list[Any]:
        ...

    @staticmethod
    def _convert_type_to_lower(values: dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
        if isinstance(values, list):
            return [
                AiPrompt._convert_type_to_lower(v) if isinstance(v, (dict, list)) else v
                for v in values]
        if isinstance(values, dict):
            return {
                k: (
                    AiPrompt._convert_type_to_lower(v) if isinstance(v, (dict, list)) else (
                        v.lower() if k == "type" and isinstance(v, str) else v
                    )
                )
                for (k, v) in values.items()
            }

    def to_openai_messages(self) -> Iterable[ChatCompletionMessageParam]:
        function_id = AiPromptFunctionId()
        return [
            {
                "role": "system",
                "content": self.system,
            },
            *(
                list(itertools.chain.from_iterable([message.to_openai_messages(function_id) for message in self.messages[:-1]]))
            ),
            *(
                list(itertools.chain.from_iterable([message.to_openai_messages(function_id, self.rag_info) for message in self.messages[-1:]]))
            ),
        ]

    def to_openai_response_format(self) -> completion_create_params.ResponseFormat:
        if self.is_json and self.schema:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "ossans_navi_response",
                    "description": "ossans_navi_response schema",
                    "schema": AiPrompt._convert_type_to_lower(
                        self.schema.model_dump(exclude_unset=True, exclude_defaults=True)
                    ),
                },
            }
        else:
            return {
                "type": "text",
            }

    def to_openai_tools(self) -> Iterable[ChatCompletionToolParam]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_last_message_detail",
                    "description": "Get detailed information about the message.",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_related_information",
                    "description": "Get Related information found in this slack group.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "terms": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        }
                    }
                },
            },
        ]

    def to_gemini_config(self) -> types.GenerateContentConfigDict:
        return {
            "system_instruction": {"parts": [{"text": self.system}]},
            "candidate_count": self.choices,
            "response_mime_type": "application/json" if self.is_json else None,
            "response_schema": self.schema.model_dump(exclude_unset=True, exclude_defaults=True) if self.is_json and self.schema else None,
            "tool_config": {
                "function_calling_config": {"mode": types.FunctionCallingConfigMode.NONE},
            },
            "tools": [
                {
                    "function_declarations": [
                        {
                            "name": "get_last_message_detail",
                            "description": "Get detailed information about the message.",
                        },
                        {
                            "name": "get_related_information",
                            "description": "Get Related information found in this slack group.",
                            "parameters": {
                                "type": types.Type.OBJECT,
                                "properties": {
                                    "terms": {
                                        "type": types.Type.ARRAY,
                                        "items": {"type": types.Type.STRING},
                                    }
                                }
                            }
                        }
                    ]
                }
            ],
        }

    def to_gemini_contents(self) -> types.ContentListUnionDict:
        contents: list[types.ContentUnionDict] = []
        for (i, message) in enumerate(self.messages):
            if i + 1 != len(self.messages):
                # ループの最後以外のメッセージには RAG 情報を付与しない
                contents.extend(message.to_gemini_content())
            else:
                # ループの最後のメッセージだけ RAG 情報を付与する
                contents.extend(message.to_gemini_content(self.rag_info))
        return contents

    def to_gemini_rest(self) -> dict[str, Any]:
        gemini_config = self.to_gemini_config()
        return {
            "generation_config": {
                k: v for (k, v) in gemini_config.items() if k in ("candidate_count", "response_mime_type", "response_schema")
            },
            **(
                {
                    k: v for (k, v) in gemini_config.items() if k in ("system_instruction", "tool_config", "tools")
                }
            ),
            "contents": AiPrompt.convert_bytes_to_base64(self.to_gemini_contents()),
        }

    @staticmethod
    def convert_bytes_to_base64(value: dict[str, Any] | list[Any] | Any) -> dict[str, Any] | list[Any] | Any:
        if isinstance(value, dict):
            return {k: AiPrompt.convert_bytes_to_base64(v) for k, v in value.items()}
        if isinstance(value, list):
            return [AiPrompt.convert_bytes_to_base64(v) for v in value]
        if isinstance(value, bytes):
            return base64.b64encode(value).decode()
        return value


@dataclasses.dataclass
class AiResponseMessage:
    content: str | dict[str, Any]
    role: AiPromptRole
    images: list[ossans_navi_types.Image] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class AiResponse:
    choices: list[AiResponseMessage]

    @staticmethod
    def from_openai_response(response: ChatCompletion, is_json: bool) -> 'AiResponse':
        # response の形式
        #   { "choices": [ { "message": { "content": "encoded_json_contents", "role": "assistant" }, "other_key": "some_value" } ] }
        # この中の encoded_json_contents から slack_search_words を取り出す処理
        return AiResponse([
            AiResponseMessage(
                content=json.loads(v) if is_json else v,
                role=AiPromptRole.ASSISTANT,
            )
            for choice in response.choices if isinstance(v := choice.message.content, str)
        ])

    @staticmethod
    def from_gemini_response(response: types.GenerateContentResponse, is_json: bool) -> 'AiResponse':
        ai_response_messages: list[AiResponseMessage] = []
        for candidate in (response.candidates if response.candidates else []):
            if (
                candidate.content
                and candidate.content.parts
            ):
                texts: list[str] = []
                images: list[ossans_navi_types.Image] = []
                for part in candidate.content.parts:
                    if isinstance(part.text, str):
                        if is_json:
                            ai_response_messages.append(
                                AiResponseMessage(
                                    content=json.loads(part.text),
                                    role=AiPromptRole.ASSISTANT,
                                )
                            )
                            # JSON の場合は複数の part が返ってくることを想定しない
                            # そういう場合も仕様上あるのかもしれないが不明
                            break
                        else:
                            texts.append(part.text)
                    elif (
                        isinstance(part.inline_data, dict)
                        and part.inline_data.mime_type
                        and part.inline_data.mime_type.startswith("image/")
                        and part.inline_data.data
                    ):
                        images.append(ossans_navi_types.Image(part.inline_data.data, part.inline_data.mime_type))
                if len(texts) > 0:
                    ai_response_messages.append(
                        AiResponseMessage(
                            content="\n".join(texts),
                            role=AiPromptRole.ASSISTANT,
                            images=images,
                        )
                    )
        return AiResponse(ai_response_messages)


@dataclasses.dataclass
class RefineResponse:
    permalinks: list[str]
    get_next_messages: list[str]
    get_messages: list[str]
    additional_search_words: list[str]


@dataclasses.dataclass
class QualityCheckResponse:
    user_intent: Optional[str]
    response_quality: bool


class AiService:
    def __init__(self) -> None:
        match config.AI_SERVICE_TYPE:
            case AiServiceType.OPENAI:
                self.client_openai = OpenAI(api_key=config.OPENAI_API_KEY)
            case AiServiceType.AZURE_OPENAI:
                self.client_openai = AzureOpenAI(
                    api_key=config.AZURE_OPENAI_API_KEY,
                    api_version=config.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                )
            case AiServiceType.GEMINI:
                self.client_gemini = genai.Client(api_key=config.GEMINI_API_KEY)
            case _:
                raise NotImplementedError("Unknown AiServiceType")

    def _chat_completions(
            self,
            model: AiModel,
            prompt: AiPrompt,
    ) -> AiResponse:
        if model.ai_service_type in (AiServiceType.OPENAI, AiServiceType.AZURE_OPENAI):
            return self._chat_completions_openai(model, prompt)
        elif model.ai_service_type == AiServiceType.GEMINI:
            return self._chat_completions_gemini(model, prompt)
        else:
            raise NotImplementedError(f"Unknown AiServiceType: {model.ai_service_type}")

    def _chat_completions_gemini(
            self,
            model: AiModel,
            prompt: AiPrompt,
    ) -> AiResponse:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"chat_completions({model.name}) body={json.dumps(prompt.to_gemini_rest(), ensure_ascii=False)}")
        else:
            logger.info(
                f"chat_completions({model.name}) body(shrink)={json.dumps(shrink_message(prompt.to_gemini_rest()), ensure_ascii=False)}"
            )
        start_time = time.time()
        last_exception: Optional[Exception] = None
        response: Optional[types.GenerateContentResponse] = None
        for _ in range(10):
            try:
                response = self.client_gemini.models.generate_content(
                    model=model.name,
                    config=prompt.to_gemini_config(),
                    contents=prompt.to_gemini_contents(),
                )
                break
            except Exception as e:
                # しばらく待ってからリトライする
                # choices>=2 の場合にエラーが発生するケースがあるのでリトライは choices=1 とする
                prompt.choices = 1
                logger.error(e)
                last_exception = e
                time.sleep(30)

        if response is None:
            # 実行に失敗していた場合は例外を送出
            raise last_exception or RuntimeError()
        logger.info(f"elapsed: {time.time() - start_time}")
        logger.info("response=" + str(response))
        # 利用したトークン数を加算する
        if response.usage_metadata:
            if isinstance(response.usage_metadata.prompt_token_count, int):
                model.tokens_in += response.usage_metadata.prompt_token_count
            if isinstance(response.usage_metadata.candidates_token_count, int):
                model.tokens_out += response.usage_metadata.candidates_token_count
        if not response.candidates:
            # 応答がない場合は例外を送出
            logger.error("Error empty choices, response=" + str(response))
            raise last_exception or RuntimeError()
        return AiResponse.from_gemini_response(response, prompt.is_json)

    def _chat_completions_openai(
            self,
            model: AiModel,
            prompt: AiPrompt,
    ) -> AiResponse:
        messages: Iterable = prompt.to_openai_messages()
        response_format = prompt.to_openai_response_format()
        tools = prompt.to_openai_tools()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"chat_completions({model.name}) body={
                    json.dumps(
                        {
                            "response_format": response_format,
                            "tools": tools,
                            "tool_choice": "none",
                            "messages": messages,
                        },
                        ensure_ascii=False
                    )
                }"
            )
        else:
            logger.info(f"chat_completions({model.name}) body(shrink)={json.dumps(shrink_message(messages), ensure_ascii=False)}")
        start_time = time.time()
        last_exception: Optional[Exception] = None
        response = None
        for _ in range(10):
            try:
                response = self.client_openai.chat.completions.create(
                    model=model.name,
                    response_format=response_format,
                    messages=messages,
                    n=1,  # 2025-06-05現在、OpenAI の API で n >= 2 を指定すると出力結果が途中で途切れる現象が発生するため n: 1 とする
                    timeout=300,
                    tools=tools,
                    tool_choice="none"
                )
                break
            except RateLimitError as e:
                # RateLimit の場合はしばらく待ってからリトライする
                logger.info(e)
                last_exception = e
                time.sleep(30)
            except InternalServerError as e:
                if e.message == "no healthy upstream":
                    # API の一時的なエラーの場合はわずかに待ってからリトライする
                    logger.info(e)
                    last_exception = e
                    time.sleep(2)
                    continue
                logger.error(e)
                raise e
            except Exception as e:
                logger.error(e)
                raise e

        if response is None:
            # 実行に失敗していた場合は例外を送出
            raise last_exception or RuntimeError()
        logger.info(f"elapsed: {time.time() - start_time}")
        logger.info("response=" + str(response))
        # 利用したトークン数を加算する
        if response.usage:
            model.tokens_in += response.usage.prompt_tokens
            model.tokens_out += response.usage.completion_tokens
        if not response.choices:
            # 応答がない場合は例外を送出
            logger.error("Error empty choices, response=" + str(response))
            raise last_exception or RuntimeError()
        return AiResponse.from_openai_response(response, prompt.is_json)

    def request_classify(self, model: AiModel, prompt: AiPrompt) -> dict[str, str | list[str]]:
        str_columns = ("user_intent", "user_intentions_type", "who_to_talk_to", "user_emotions",)
        list_columns = ("required_knowledge_types", "slack_emoji_names",)
        prompt.choices = 5
        response = self._chat_completions(model, prompt)
        summary: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
        for message in response.choices:
            if isinstance(message.content, dict):
                for name in (*str_columns, *list_columns):
                    if isinstance((v := message.content.get(name)), str):
                        summary[name][v] = summary[name][v] + 1
                for name in list_columns:
                    if isinstance((v := message.content.get(name)), list):
                        for w in v:
                            summary[name][w] = summary[name][w] + 1
        summary_sorted: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
        for name in (*str_columns, *list_columns):
            summary_sorted[name] = sorted(summary[name].items(), key=lambda v: (v[1], len(v[0]), v[0]), reverse=True)
        logger.info(f"classify: {summary_sorted}")
        return {
            **(
                {
                    name: v[0][0] if len(v := summary_sorted[name]) > 0 else ""
                    for name in str_columns
                }
            ),
            **(
                {
                    name: list(map(itemgetter(0), v)) if len(v := summary_sorted[name]) > 0 else []
                    for name in list_columns
                }
            )
        }

    def request_image_description(self, model: AiModel, prompt: AiPrompt) -> dict[str, list[dict[str, str]]]:
        response = self._chat_completions(model, prompt)
        message = response.choices[0]
        if not isinstance(message.content, dict):
            return {}
        content: dict[str, list[dict[str, str]]] = message.content
        if (
            isinstance(content, dict)
            and sum([1 for permalink in content.keys() if isinstance(permalink, str)]) == len(content)
            and sum([1 for analyzed_images in content.values() if isinstance(analyzed_images, list)]) == len(content)
            and sum(
                [
                    1 for v in itertools.chain.from_iterable(content.values())
                    if isinstance(v, dict) and isinstance(v.get("description"), str) and isinstance(v.get("text"), str)
                ]
            ) == len(list(itertools.chain.from_iterable(content.values())))
        ):
            return content
        else:
            return {}

    def request_slack_search_words(self, model: AiModel, prompt: AiPrompt) -> list[str]:
        # Gemini は n=2 で十分なバリエーションを生成してくれる
        prompt.choices = 2 if model.ai_service_type == AiServiceType.GEMINI else 5
        response = self._chat_completions(model, prompt)
        slack_search_words: list[str] = list(set(
            itertools.chain.from_iterable(
                [
                    [slack_search_words for slack_search_words in message.content['slack_search_words']]
                    for message in response.choices if 'slack_search_words' in message.content if isinstance(message.content, dict)
                ]
            )
        ))
        # もう一度、問い合わせる可能性があるので今回の返答をセッションに積む
        prompt.messages.append(
            AiPromptMessage(
                role=AiPromptRole.ASSISTANT,
                content=AiPromptContent(
                    data=(json.dumps(v, ensure_ascii=False, separators=(',', ':')) if isinstance(v := response.choices[0].content, dict) else "")
                )
            )
        )
        return slack_search_words

    @staticmethod
    def _analyze_refine_slack_searches_response(response: AiResponse) -> list[RefineResponse]:
        return [
            RefineResponse(
                [v for v in message.content.get('permalinks', []) if isinstance(v, str)],
                [v for v in message.content.get('get_next_messages', []) if isinstance(v, str)],
                [v for v in message.content.get('get_messages', []) if isinstance(v, str)],
                [v for v in message.content.get('additional_search_words', []) if isinstance(v, str)],
            )
            for message in response.choices if (
                isinstance(message.content, dict)
                and isinstance(message.content.get('permalinks', []), list)
                and isinstance(message.content.get('get_next_messages', []), list)
                and isinstance(message.content.get('get_messages', []), list)
                and isinstance(message.content.get('additional_search_words', []), list)
            )
        ]

    def request_refine_slack_searches(self, model: AiModel, prompt: AiPrompt) -> list[RefineResponse]:
        # Gemini は n=2 で十分な網羅性がある
        prompt.choices = 2 if model.ai_service_type == AiServiceType.GEMINI else 5
        response = self._chat_completions(model, prompt)
        return AiService._analyze_refine_slack_searches_response(response)

    @staticmethod
    def _analyze_lastshot_response(response: AiResponse) -> list[ossans_navi_types.LastshotResponse]:
        return [
            ossans_navi_types.LastshotResponse(message.content, message.images)
            for message in response.choices if isinstance(message.content, str)
        ]

    def request_lastshot(self, model: AiModel, prompt: AiPrompt, n: int = 1) -> list[ossans_navi_types.LastshotResponse]:
        # Gemini は大きいプロンプトのパターンで choice>=2 が原因となるエラーケースがある、よって choice=1 とする
        prompt.choices = 1 if model.ai_service_type == AiServiceType.GEMINI else n
        for _ in range(2):
            response = self._chat_completions(model, prompt)
            if len(result := AiService._analyze_lastshot_response(response)) > 0:
                return result
        logger.error("Error empty choices, response=" + str(response))
        raise ValueError("Error empty choices, response=" + str(response))

    @staticmethod
    def _analyze_quality_check_response(response: AiResponse) -> Optional[QualityCheckResponse]:
        result = [
            QualityCheckResponse(
                message.content['user_intent'],
                message.content['response_quality'],
            )
            for message in response.choices if (
                isinstance(message.content, dict)
                and 'user_intent' in message.content
                and 'response_quality' in message.content
                and isinstance(message.content['response_quality'], bool)
            )
        ]
        if len(result) == 0:
            return None
        return result[0]

    def request_quality_check(self, model: AiModel, prompt: AiPrompt) -> QualityCheckResponse:
        for _ in range(2):
            response = self._chat_completions(model, prompt)
            if (result := AiService._analyze_quality_check_response(response)):
                return result
        logger.error("Error empty choices, response=" + str(response))
        raise ValueError("Error empty choices, response=" + str(response))
