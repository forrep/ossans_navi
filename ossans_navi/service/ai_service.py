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
from typing import Any, Iterable, Optional

from google import genai
from google.genai import types
from openai import NOT_GIVEN, AzureOpenAI, InternalServerError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion

from ossans_navi import config
from ossans_navi.common.logger import shrink_message
from ossans_navi.config import AiServiceType
from ossans_navi.service.ai_tokenize_service import AiTokenize, AiTokenizeGpt4o

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
class AiPromptImage:
    data: bytes

    @property
    def image_uri(self) -> str:
        return f"data:image/png;base64,{base64.b64encode(self.data).decode()}"


@dataclasses.dataclass
class AiPromptContent:
    text: str
    images: list[AiPromptImage] = dataclasses.field(default_factory=list)

    def to_openai_prompt(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "text",
                "text": self.text,
            },
            *[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image.image_uri
                    }
                }
                for image in self.images
            ]
        ]

    def to_gemini_prompt(self) -> list[types.Part]:
        return [
            types.Part.from_text(text=self.text),
            *[types.Part.from_bytes(data=image.data, mime_type="image/png") for image in self.images]
        ]

    def to_gemini_dict(self) -> list[dict[str, Any]]:
        return [
            {"text": self.text},
            *[
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(image.data).decode(),
                    }
                }
                for image in self.images
            ]
        ]


@dataclasses.dataclass
class AiPromptMessage:
    role: AiPromptRole
    content: AiPromptContent
    name: Optional[str] = dataclasses.field(default=None)

    def to_openai_prompt(self) -> dict[str, str | list[dict[str, Any]]]:
        return {
            "role": self.role.value,
            "content": self.content.to_openai_prompt(),
            **({"name": self.name} if self.name else {})
        }

    def to_gemini_prompt(self) -> types.Content:
        return types.Content(
            # Gemini は user, model の2種類
            role="model" if self.role == AiPromptRole.ASSISTANT else self.role.value,
            parts=self.content.to_gemini_prompt()
        )

    def to_gemini_dict(self) -> dict[str, Any]:
        return {
            "role": "model" if self.role == AiPromptRole.ASSISTANT else self.role.value,
            "parts": self.content.to_gemini_dict(),
        }


@dataclasses.dataclass
class AiPrompt:
    system: str
    messages: list[AiPromptMessage]
    schema: Optional[types.Schema] = dataclasses.field(default=None)
    choices: int = dataclasses.field(default=1)

    def to_openai_prompt(self) -> list[dict[str, str | list[dict[str, Any]]]]:
        return [
            {
                "role": "system",
                "content": self.system,
            },
            *[
                message.to_openai_prompt() for message in self.messages
            ],
        ]

    def to_gemini_prompt(self) -> list[types.Content]:
        return [
            message.to_gemini_prompt() for message in self.messages
        ]

    def to_gemini_system_prompt(self) -> types.Content:
        return types.Content(parts=[types.Part.from_text(text=self.system)])

    def to_gemini_dict(self) -> dict[str, Any]:
        return {
            "generationConfig": {
                "candidateCount": self.choices,
                **(
                    {
                        "responseMimeType": "application/json",
                        "responseSchema": self.schema.model_dump(exclude_unset=True, exclude_defaults=True),
                    }
                    if self.schema else {}
                )
            },
            "systemInstruction": {
                "parts": [
                    {
                        "text": self.system,
                    }
                ]
            },
            "contents": [message.to_gemini_dict() for message in self.messages]
        }


@dataclasses.dataclass
class AiResponseMessage:
    content: str | dict[str, Any]
    role: AiPromptRole


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
    def str_or_json_content(json_string: str) -> str:
        try:
            if isinstance((v := json.loads(json_string)), dict) and isinstance((w := v.get("content")), str):
                return w
            else:
                return json_string
        except json.JSONDecodeError:
            return json_string

    @staticmethod
    def from_gemini_response(response: types.GenerateContentResponse, is_json: bool) -> 'AiResponse':
        return AiResponse([
            AiResponseMessage(
                # Gemini は間違えて JSON を返すことがあるので、その場合はパースして content キーを取り出す
                content=json.loads(v) if is_json else AiResponse.str_or_json_content(v),
                role=AiPromptRole.ASSISTANT,
            )
            for candidate in response.candidates if isinstance(v := candidate.content.parts[0].text, str)
        ])


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
                raise NotImplementedError(f"Unknown AiServiceType: {config.AI_SERVICE_TYPE}")

    def _chat_completions(
            self,
            model: AiModel,
            prompt: AiPrompt,
            is_json: bool = True,
    ) -> AiResponse:
        if model.ai_service_type in (AiServiceType.OPENAI, AiServiceType.AZURE_OPENAI):
            return self._chat_completions_openai(
                model,
                prompt,
                is_json,
            )
        elif model.ai_service_type == AiServiceType.GEMINI:
            return self._chat_completions_gemini(
                model,
                prompt,
                is_json,
            )
        else:
            raise NotImplementedError(f"Unknown AiServiceType: {model.ai_service_type}")

    def _chat_completions_gemini(
            self,
            model: AiModel,
            prompt: AiPrompt,
            is_json: bool,
    ) -> AiResponse:
        messages: Iterable = prompt.to_gemini_prompt()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"_chat_completions model={model.name}, prompt={json.dumps(prompt.to_gemini_dict(), ensure_ascii=False)}")
        else:
            logger.info(
                f"_chat_completions model={model.name}, prompt(shrink)={json.dumps(shrink_message(prompt.to_gemini_dict()), ensure_ascii=False)}"
            )
        start_time = time.time()
        last_exception: Optional[Exception] = None
        response: types.GenerateContentResponse = None
        for _ in range(10):
            try:
                response = self.client_gemini.models.generate_content(
                    model=model.name,
                    config=types.GenerateContentConfig(
                        system_instruction=prompt.to_gemini_system_prompt(),
                        candidate_count=prompt.choices,
                        response_mime_type="application/json" if is_json else None,
                        response_schema=prompt.schema if is_json else None,
                    ),
                    contents=messages,
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
            model.tokens_in += response.usage_metadata.prompt_token_count
            model.tokens_out += response.usage_metadata.candidates_token_count
        if not response.candidates:
            # 応答がない場合は例外を送出
            logger.error("Error empty choices, response=" + str(response))
            raise last_exception or RuntimeError()
        return AiResponse.from_gemini_response(response, is_json)

    def _chat_completions_openai(
            self,
            model: AiModel,
            prompt: AiPrompt,
            is_json: bool,
    ) -> AiResponse:
        messages: Iterable = prompt.to_openai_prompt()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"_chat_completions model={model.name}, messages={json.dumps(messages, ensure_ascii=False)}")
        else:
            logger.info(f"_chat_completions model={model.name}, messages(shrink)={json.dumps(shrink_message(messages), ensure_ascii=False)}")
        start_time = time.time()
        last_exception: Optional[Exception] = None
        response = None
        for _ in range(10):
            try:
                response = self.client_openai.chat.completions.create(
                    model=model.name,
                    response_format=({"type": "json_object"} if is_json else NOT_GIVEN),
                    messages=messages,
                    n=prompt.choices,
                    timeout=300,
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
        return AiResponse.from_openai_response(response, is_json)

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
                content=AiPromptContent(text=json.dumps(v if (v := response.choices[0].content) else "{}", ensure_ascii=False))
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
    def _analyze_lastshot_response(response: AiResponse) -> list[str]:
        return [message.content for message in response.choices if isinstance(message.content, str)]

    def request_lastshot(self, model: AiModel, prompt: AiPrompt, n: int = 1) -> list[str]:
        # Gemini は大きいプロンプトのパターンで choice>=2 が原因となるエラーケースがある、よって choice=1 とする
        prompt.choices = 1 if model.ai_service_type == AiServiceType.GEMINI else n
        for _ in range(2):
            response = self._chat_completions(model, prompt, False)
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
