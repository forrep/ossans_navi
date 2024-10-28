import dataclasses
import itertools
import json
import logging
import threading
import time
from typing import Optional

from openai import (NOT_GIVEN, AzureOpenAI, InternalServerError, OpenAI,
                    RateLimitError)
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
    cost_in: float
    cost_out: float
    tokenizer: AiTokenize
    tokens_in: int = dataclasses.field(default=0, init=False)
    tokens_out: int = dataclasses.field(default=0, init=False)

    def get_total_cost(self) -> float:
        return round(self.cost_in * self.tokens_in / 1_000_000 + self.cost_out * self.tokens_out / 1_000_000, 2)


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
                    config.OPENAI_MODEL_LOW_COST_IN,
                    config.OPENAI_MODEL_LOW_COST_OUT,
                    AiTokenizeGpt4o
                )
                models.high_quality = AiModel(
                    config.OPENAI_MODEL_HIGH_QUALITY,
                    config.OPENAI_MODEL_HIGH_QUALITY_IN,
                    config.OPENAI_MODEL_HIGH_QUALITY_OUT,
                    AiTokenizeGpt4o
                )
            case AiServiceType.AZURE_OPENAI:
                models.low_cost = AiModel(
                    config.AZURE_OPENAI_MODEL_LOW_COST,
                    config.AZURE_OPENAI_MODEL_LOW_COST_IN,
                    config.AZURE_OPENAI_MODEL_LOW_COST_OUT,
                    AiTokenizeGpt4o
                )
                models.high_quality = AiModel(
                    config.AZURE_OPENAI_MODEL_HIGH_QUALITY,
                    config.AZURE_OPENAI_MODEL_HIGH_QUALITY_IN,
                    config.AZURE_OPENAI_MODEL_HIGH_QUALITY_OUT,
                    AiTokenizeGpt4o
                )
            case _:
                raise NotImplementedError(f"Unknown Service: {config.AiServiceType}")
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
        return round(sum([model.get_total_cost() for model in self.models()]), 4)


@dataclasses.dataclass
class AiServiceSession:
    messages: list

    def append_message(self, message: dict) -> None:
        self.messages.append(message)


@dataclasses.dataclass
class RefineResponse:
    permalinks: list[str]
    get_next_messages: list[str]
    get_messages: list[str]
    additional_search_words: list[str]


@dataclasses.dataclass
class LastshotResponse:
    user_intent: str
    response_message: str
    confirm_message: str | None
    response_quality: bool


class AiService:
    def __init__(self) -> None:
        match config.AI_SERVICE_TYPE:
            case AiServiceType.OPENAI:
                self.client = OpenAI(
                    api_key=config.OPENAI_API_KEY,
                )
            case AiServiceType.AZURE_OPENAI:
                self.client = AzureOpenAI(
                    api_key=config.AZURE_OPENAI_API_KEY,
                    api_version=config.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                )
            case _:
                raise NotImplementedError(f"Unknown Service: {config.AiServiceType}")

    def _chat_completions(
            self,
            model: AiModel,
            messages: list[dict],
            tools: list[dict] = NOT_GIVEN,
            tool_choice: str = NOT_GIVEN,
            n: int = NOT_GIVEN,
            max_tokens: int = NOT_GIVEN
    ) -> ChatCompletion:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"_chat_completions model={model.name}, messages=" + json.dumps(messages, ensure_ascii=False))
        else:
            logger.info(f"_chat_completions model={model.name}, messages(shrink)=" + json.dumps(shrink_message(messages), ensure_ascii=False))
        start_time = time.time()
        last_exception: Optional[Exception]
        for _ in range(10):
            try:
                response = self.client.chat.completions.create(
                    model=model.name,
                    response_format={"type": "json_object"},
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    n=n,
                    max_tokens=max_tokens,
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
            raise last_exception
        logger.info(f"elapsed: {time.time() - start_time}")
        logger.info("response=" + str(response))
        # 利用したトークン数を加算する
        model.tokens_in += response.usage.prompt_tokens
        model.tokens_out += response.usage.completion_tokens
        return response

    def request_classification(self, model: AiModel, messages: list) -> tuple[str, str]:
        response = self._chat_completions(model, messages, n=5)
        messages = [{k: (json.loads(v) if k == 'content' else v) for (k, v) in choice.message} for choice in response.choices]
        if len(messages) == 0:
            logger.error("Error empty choices, response=" + str(response))
            raise ValueError()
        message_type = {}
        slack_emoji_name = {}
        for message in messages:
            if (
                'content' in message
                and 'message_type' in message['content']
                and isinstance(message['content']['message_type'], str)
            ):
                message_type[message['content']['message_type']] = message_type.get(message['content']['message_type'], 0) + 1
                if "slack_emoji_name" in message['content'] and len(message['content']["slack_emoji_name"]) > 0:
                    slack_emoji_name[message['content']["slack_emoji_name"]] = slack_emoji_name.get(message['content']["slack_emoji_name"], 0) + 1
        sorted_message_type: list[tuple[str, int]] = sorted(message_type.items(), key=lambda v: (v[1], v[0]), reverse=True)
        sorted_slack_emoji_name: list[tuple[str, int]] = sorted(slack_emoji_name.items(), key=lambda v: (v[1], v[0]), reverse=True)
        decided_message_type = "other"
        decided_slack_emoji_name = ""
        if len(sorted_message_type) > 0 and sorted_message_type[0][1] >= 2:
            decided_message_type = sorted_message_type[0][0]
        if len(sorted_slack_emoji_name) > 0:
            decided_slack_emoji_name = sorted_slack_emoji_name[0][0]
        return (decided_message_type, decided_slack_emoji_name)

    def request_image_description(self, model: AiModel, messages: list) -> dict[str, list[dict[str, str]]]:
        response = self._chat_completions(model, messages)
        messages = [{k: (json.loads(v) if k == 'content' else v) for (k, v) in choice.message} for choice in response.choices]
        if len(messages) == 0:
            logger.error("Error empty choices, response=" + str(response))
            raise ValueError()
        message: dict = messages[0]
        if 'content' not in message:
            return {}
        content: dict[str, list[dict[str, str]]] = message['content']
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

    def request_slack_search_words(self, model: AiModel, session: AiServiceSession) -> list[str]:
        response = self._chat_completions(model, session.messages, n=5)
        if len(response.choices) == 0:
            logger.error("Error empty choices, response=" + str(response))
            raise ValueError()
        # response の形式
        #   { "choices": [ { "message": { "content": "encoded_json_contents", "role": "assistant" }, "other_key": "some_value" } ] }
        # この中の encoded_json_contents から slack_search_words を取り出す処理
        messages = [{k: (json.loads(v) if k == 'content' else v) for (k, v) in choice.message} for choice in response.choices]
        slack_search_words: list[str] = list(set(
            itertools.chain.from_iterable(
                [
                    [slack_search_words for slack_search_words in message['content']['slack_search_words']]
                    for message in messages if 'content' in message and 'slack_search_words' in message['content']
                ]
            )
        ))
        # もう一度、問い合わせる可能性があるので今回の返答をセッションに積む、型を dict へ変換
        session.messages.append({k: v for (k, v) in response.choices[0].message if k in ("content", "role")})
        return slack_search_words

    @staticmethod
    def _analyze_refine_slack_searches_response(response: ChatCompletion) -> list[RefineResponse]:
        messages: list[dict[str, dict]] = []
        for choice in response.choices:
            try:
                # JSON がパースできない場合もある、そのときは無視する
                messages.append(
                    {k: (json.loads(v) if k == 'content' else v) for (k, v) in choice.message}
                )
            except json.JSONDecodeError:
                pass
        return [
            RefineResponse(
                [v for v in message['content'].get('permalinks', []) if isinstance(v, str)],
                [v for v in message['content'].get('get_next_messages', []) if isinstance(v, str)],
                [v for v in message['content'].get('get_messages', []) if isinstance(v, str)],
                [v for v in message['content'].get('additional_search_words', []) if isinstance(v, str)],
            )
            for message in messages if (
                'content' in message
                and isinstance(message['content'].get('permalinks', []), list)
                and isinstance(message['content'].get('get_next_messages', []), list)
                and isinstance(message['content'].get('get_messages', []), list)
                and isinstance(message['content'].get('additional_search_words', []), list)
            )
        ]

    def request_refine_slack_searches(self, model: AiModel, messages: list) -> list[RefineResponse]:
        response = self._chat_completions(model, messages, n=5)
        if len(response.choices) == 0:
            logger.error("Error empty choices, response=" + str(response))
            raise ValueError()
        return AiService._analyze_refine_slack_searches_response(response)

    @staticmethod
    def _analyze_lastshot_response(response: ChatCompletion) -> list[LastshotResponse]:
        messages = [{k: (json.loads(v) if k == 'content' else v) for (k, v) in choice.message} for choice in response.choices]
        return [
            LastshotResponse(
                message['content']['user_intent'],
                message['content']['response_message'],
                message['content']['confirm_message'] if isinstance(message['content']['confirm_message'], str) else None,
                message['content']['response_quality'],
            )
            for message in messages if (
                'content' in message
                and 'user_intent' in message['content']
                and 'response_message' in message['content']
                and 'confirm_message' in message['content']
                and 'response_quality' in message['content']
                and isinstance(message['content']['response_message'], str)
                and len(message['content']['response_message']) > 0
                and isinstance(message['content']['response_quality'], bool)
            )
        ]

    def request_lastshot(self, model: AiModel, messages: list) -> list[LastshotResponse]:
        for _ in range(2):
            response = self._chat_completions(model, messages)
            if len(result := AiService._analyze_lastshot_response(response)) > 0:
                return result
        logger.error("Error empty choices, response=" + str(response))
        raise ValueError("Error empty choices, response=" + str(response))
