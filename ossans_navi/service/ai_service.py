import asyncio
import base64
import itertools
import json
import logging
import time
from collections import defaultdict
from enum import Enum
from io import BytesIO
from operator import itemgetter
from typing import Any, Iterable, Optional, overload

from google import genai
from google.genai import types
from openai import AsyncAzureOpenAI, AsyncOpenAI, InternalServerError, RateLimitError
from openai.types.chat import (ChatCompletion, ChatCompletionContentPartParam, ChatCompletionMessageParam, ChatCompletionToolParam,
                               completion_create_params)
from pydantic import BaseModel, Field

from ossans_navi import config
from ossans_navi.common import async_utils
from ossans_navi.common.logger import shrink_message
from ossans_navi.type import ossans_navi_type
from ossans_navi.type.ai_type import AiModelsUsage, AiModelUsage, AiPromptSlackMessage, AiServiceType

logger = logging.getLogger(__name__)


class AiPromptRole(Enum):
    ASSISTANT = ("model", "assistant")
    USER = ("user", "user")

    def __init__(self, gemini_role, openai_role) -> None:
        self.gemini_role = gemini_role
        self.openai_role = openai_role


class AiPromptRagInfo(BaseModel):
    contents: list[Any] | dict[str, Any]
    words: list[str]


class AiPromptUploadFile(BaseModel):
    data: bytes
    mimetype: str
    title: str
    file: Optional[types.File] = Field(default=None, init=False)

    def to_bytes_io(self) -> BytesIO:
        return BytesIO(self.data)

    @property
    def image_uri(self) -> str:
        return f"data:image/png;base64,{base64.b64encode(self.data).decode()}"


class AiPromptContent(BaseModel):
    data: str | AiPromptSlackMessage
    images: list[AiPromptUploadFile] = Field(default_factory=list)
    videos: list[AiPromptUploadFile] = Field(default_factory=list)
    audios: list[AiPromptUploadFile] = Field(default_factory=list)

    @property
    def is_dict(self) -> bool:
        return isinstance(self.data, AiPromptSlackMessage)

    @property
    def text(self) -> str:
        if isinstance(self.data, AiPromptSlackMessage):
            return self.data.content
        else:
            return self.data

    @property
    def detail(self) -> dict[str, Any]:
        if isinstance(self.data, AiPromptSlackMessage):
            return {k: v for (k, v) in self.data.model_dump().items() if k != "content"}
        else:
            return {}


class AiPromptFunctionId(BaseModel):
    call: int = Field(default=0, init=False)
    response: int = Field(default=0, init=False)

    @property
    def call_id(self) -> int:
        self.call = self.call + 1
        return self.call

    @property
    def response_id(self) -> int:
        self.response = self.response + 1
        return self.response


class AiPromptMessage(BaseModel):
    role: AiPromptRole
    content: AiPromptContent
    name: Optional[str] = Field(default=None)

    def to_openai_messages(self, function_id: AiPromptFunctionId, rag_info: Optional[AiPromptRagInfo] = None) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        message: Optional[ChatCompletionMessageParam] = None
        if self.role == AiPromptRole.ASSISTANT:
            message = {
                "role": AiPromptRole.ASSISTANT.openai_role,
                "content": self.content.text,
            }
        elif self.role == AiPromptRole.USER:
            # USERロールの場合のみ画像や <metadata> を入力する
            # OpenAI は ASSISTANTロールでの画像を入力に未対応、誤って入力するとエラーとなる
            # 一方で ASSISTANT の応答文に画像リンクが含まれると Slack 上ではメッセージに画像が添付されるため、
            # ASSISTANT ロールのメッセージにも画像が添付されるケースはある
            # <metadata> を ASSISTANTロールで入力してしまうと、ハルシネーションで架空の <metadata> を生成してしまうことがあるため入力してはいけない
            message_contents: list[ChatCompletionContentPartParam] = []
            message_contents.append(
                {
                    "type": "text",
                    "text": (
                        self.content.text
                        + (
                            (
                                "\n\n<metadata>"
                                + json.dumps(self.content.detail, ensure_ascii=False, separators=(',', ':'))
                                + "\n</metadata>"
                            ) if self.content.is_dict else ""
                        )
                    ),
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
                "role": AiPromptRole.USER.openai_role,
                "content": message_contents,
            }
        else:
            raise ValueError("Unknown AiPromptRole: " + str(self.role))
        if self.name:
            message["name"] = self.name
        messages.append(message)

        if rag_info:
            messages.append(
                {
                    "role": AiPromptRole.ASSISTANT.openai_role,
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{function_id.call_id}",
                            "type": "function",
                            "function": {
                                "name": "get_related_information",
                                "arguments": json.dumps(rag_info.words, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
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
        if self.role == AiPromptRole.ASSISTANT:
            parts.append(
                {
                    "text": self.content.text
                }
            )
        elif self.role == AiPromptRole.USER:
            # USERロールの場合のみ画像・映像・音声や <metadata> を入力する
            # Gemini は ASSISTANTロールで画像を入力してもエラーにはならないが処理対象にもならない、無駄なので送らない
            # ASSISTANT の応答文に画像リンクが含まれると Slack 上ではメッセージに画像が添付されるため、
            # ASSISTANT ロールのメッセージにも画像が添付されるケースはある
            # <metadata> を ASSISTANTロールで入力してしまうと、ハルシネーションで架空の <metadata> を生成してしまうことがあるため入力してはいけない
            parts.append(
                {
                    "text": (
                        self.content.text
                        + (
                            # <metadata> が存在する場合を入力する（self.content.is_dict の場合）
                            (
                                "\n\n<metadata>\n"
                                + json.dumps(self.content.detail, ensure_ascii=False, separators=(',', ':'))
                                + "\n</metadata>"
                            ) if self.content.is_dict else ""
                        )
                    )
                }
            )
            for image in self.content.images:
                if image.file is not None:
                    parts.append({"text": f"Attachment(below): {image.title}"})
                    parts.append(
                        {
                            "file_data": {
                                "mime_type": image.file.mime_type,
                                "file_uri": image.file.uri,
                            }
                        }
                    )
            for video in self.content.videos:
                if video.file is not None:
                    parts.append({"text": f"Attachment(below): {video.title}"})
                    parts.append(
                        {
                            "file_data": {
                                "mime_type": video.file.mime_type,
                                "file_uri": video.file.uri,
                            },
                            "video_metadata": {
                                "fps": config.VIDEO_FPS,
                            }
                        }
                    )
            for audio in self.content.audios:
                if audio.file is not None:
                    parts.append({"text": f"Attachment(below): {audio.title}"})
                    parts.append(
                        {
                            "file_data": {
                                "mime_type": audio.file.mime_type,
                                "file_uri": audio.file.uri,
                            },
                        }
                    )
        contents.append({
            "role": self.role.gemini_role,
            "parts": parts,
        })

        if rag_info:
            contents.append(
                {
                    "role": AiPromptRole.ASSISTANT.gemini_role,
                    "parts": [
                        {
                            "function_call": {
                                "name": "get_related_information",
                                "args": {
                                    "terms": rag_info.words
                                },
                            }
                        }
                    ],
                }
            )
            contents.append(
                {
                    "role": AiPromptRole.USER.gemini_role,
                    "parts": [
                        {
                            "function_response": {
                                "name": "get_related_information",
                                "response": {
                                    **(
                                        {
                                            "status": (
                                                "No related information was found in the get_related_information results, "
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
                    ],
                }
            )

        return contents


class AiPrompt(BaseModel):
    system: str
    messages: list[AiPromptMessage]
    response_schema: Optional[types.Schema] = Field(default=None)
    choices: int = Field(default=1)
    rag_info: Optional[AiPromptRagInfo] = Field(default=None)
    tools_url_context: bool = Field(default=False)

    @property
    def is_json(self) -> bool:
        return self.response_schema is not None

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
        if self.is_json and self.response_schema:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "ossans_navi_response",
                    "description": "ossans_navi_response schema",
                    "schema": AiPrompt._convert_type_to_lower(
                        self.response_schema.model_dump(exclude_unset=True, exclude_defaults=True)
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
            "max_output_tokens": config.MAX_OUTPUT_TOKENS,
            "response_mime_type": "application/json" if self.is_json else None,
            "response_schema": (
                self.response_schema.model_dump(exclude_unset=True, exclude_defaults=True)
                if self.is_json and self.response_schema else None
            ),
            "tool_config": {
                "function_calling_config": {"mode": types.FunctionCallingConfigMode.NONE},
            },
            "tools": [
                *(
                    [
                        {
                            "function_declarations": [
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
                    ] if not self.tools_url_context else []
                ),
                *(
                    [{"url_context": {}}] if self.tools_url_context else []
                )
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
                k: v for (k, v) in gemini_config.items() if k in ("candidate_count", "response_mime_type", "response_schema", "max_output_tokens",)
            },
            **(
                {
                    k: v for (k, v) in gemini_config.items() if k in ("system_instruction", "tool_config", "tools")
                }
            ),
            "contents": AiPrompt.convert_bytes_to_base64(self.to_gemini_contents()),
        }

    def get_upload_files(self) -> list[AiPromptUploadFile]:
        """USERロールに紐づく画像・映像・音声ファイルを全て取得する"""
        return list(itertools.chain.from_iterable([
            message.content.images + message.content.videos + message.content.audios
            for message in self.messages if message.role == AiPromptRole.USER
        ]))

    @staticmethod
    def convert_bytes_to_base64(value: dict[str, Any] | list[Any] | Any) -> dict[str, Any] | list[Any] | Any:
        if isinstance(value, dict):
            return {k: AiPrompt.convert_bytes_to_base64(v) for k, v in value.items()}
        if isinstance(value, list):
            return [AiPrompt.convert_bytes_to_base64(v) for v in value]
        if isinstance(value, bytes):
            return base64.b64encode(value).decode()
        return value


class AiResponseMessage(BaseModel):
    content: str | dict[str, Any]
    role: AiPromptRole
    images: list[ossans_navi_type.Image] = Field(default_factory=list)


class AiResponse(BaseModel):
    choices: list[AiResponseMessage]

    @staticmethod
    def from_openai_response(response: ChatCompletion, is_json: bool) -> 'AiResponse':
        # response の形式
        #   { "choices": [ { "message": { "content": "encoded_json_contents", "role": "assistant" }, "other_key": "some_value" } ] }
        # この中の encoded_json_contents から slack_search_words を取り出す処理
        return AiResponse(
            choices=[
                AiResponseMessage(
                    content=json.loads(v) if is_json else v,
                    role=AiPromptRole.ASSISTANT,
                )
                for choice in response.choices if isinstance(v := choice.message.content, str)
            ]
        )

    @staticmethod
    def from_gemini_response(response: types.GenerateContentResponse, is_json: bool) -> 'AiResponse':
        ai_response_messages: list[AiResponseMessage] = []
        for candidate in (response.candidates if response.candidates else []):
            if candidate.content and candidate.content.parts:
                texts: list[str] = []
                images: list[ossans_navi_type.Image] = []
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
                        images.append(ossans_navi_type.Image(part.inline_data.data, part.inline_data.mime_type))
                if len(texts) > 0:
                    ai_response_messages.append(
                        AiResponseMessage(
                            content="\n".join(texts),
                            role=AiPromptRole.ASSISTANT,
                            images=images,
                        )
                    )
        return AiResponse(choices=ai_response_messages)


class RefineResponse(BaseModel):
    permalinks: list[str]
    additional_search_words: list[str]


class QualityCheckResponse(BaseModel):
    user_intent: Optional[str]
    response_quality: bool


class AiService:
    client_openai_instance: Optional[AsyncOpenAI] = None
    client_azure_openai_instance: Optional[AsyncAzureOpenAI] = None
    client_gemini_instance: Optional[genai.Client] = None

    def __init__(self):
        self.models_usage = AiModelsUsage.new()

    @classmethod
    async def start(cls) -> None:
        # 環境変数にセットされた APIキーから、それぞれのクライアントインスタンスを生成する
        if config.OPENAI_API_KEY:
            AiService.client_openai_instance = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        if config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_ENDPOINT:
            AiService.client_azure_openai_instance = AsyncAzureOpenAI(
                api_key=config.AZURE_OPENAI_API_KEY,
                api_version=config.AZURE_OPENAI_API_VERSION,
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            )
        if config.GEMINI_API_KEY:
            AiService.client_gemini_instance = genai.Client(api_key=config.GEMINI_API_KEY)

        # AiModelsUsage.new() することで環境変数に設定したモデル名が正しいことを検証するのと、
        # 利用するモデルに対応する AiService.client_*_instance が None でないことを検証する
        models_usage = AiModelsUsage.new()
        for model in models_usage.models:
            match model.ai_service_type:
                case AiServiceType.OPENAI:
                    if not AiService.client_openai_instance:
                        raise ValueError("OSN_OPENAI_API_KEY is required when using OpenAI service.")
                case AiServiceType.AZURE_OPENAI:
                    if not AiService.client_azure_openai_instance:
                        raise ValueError("OSN_AZURE_OPENAI_API_KEY and OSN_AZURE_OPENAI_ENDPOINT are required when using Azure OpenAI service.")
                case AiServiceType.GEMINI:
                    if not AiService.client_gemini_instance:
                        raise ValueError("OSN_GEMINI_API_KEY is required when using Gemini service.")
                case _:
                    raise NotImplementedError("Unknown AiServiceType")

    @property
    def client_openai(self) -> AsyncOpenAI:
        if AiService.client_openai_instance is None:
            raise RuntimeError("AiService is not started.")
        return AiService.client_openai_instance

    @property
    def client_azure_openai(self) -> AsyncAzureOpenAI:
        if AiService.client_azure_openai_instance is None:
            raise RuntimeError("AiService is not started.")
        return AiService.client_azure_openai_instance

    @property
    def client_gemini(self) -> genai.Client:
        if AiService.client_gemini_instance is None:
            raise RuntimeError("AiService is not started.")
        return AiService.client_gemini_instance

    async def _chat_completions(
        self,
        model: AiModelUsage,
        prompt: AiPrompt,
    ) -> AiResponse:
        if model.ai_service_type in (AiServiceType.OPENAI, AiServiceType.AZURE_OPENAI):
            return await self._chat_completions_openai(model, prompt)
        elif model.ai_service_type == AiServiceType.GEMINI:
            return await self._chat_completions_gemini(model, prompt)
        else:
            raise NotImplementedError(f"Unknown AiServiceType: {model.ai_service_type}")

    async def _chat_completions_gemini(
        self,
        model: AiModelUsage,
        prompt: AiPrompt,
    ) -> AiResponse:
        logger.debug(f"start _chat_completions_gemini: {str(self.client_gemini)}")
        start_time = time.time()
        last_exception: Optional[Exception] = None
        response: Optional[types.GenerateContentResponse] = None
        try:
            if len(prompt.get_upload_files()) > 0:
                logger.debug(f"Uploading files: {', '.join([file.title for file in prompt.get_upload_files()])}")
                uploaded_files = await async_utils.asyncio_gather(
                    *[
                        self.client_gemini.aio.files.upload(
                            file=file.to_bytes_io(),
                            config={
                                "mime_type": file.mimetype,
                                "display_name": file.title,
                            },
                        )
                        for file in prompt.get_upload_files()
                    ],
                    concurrency=2,
                )
                for (file, uploaded_file) in zip(prompt.get_upload_files(), uploaded_files):
                    file.file = uploaded_file
                logger.debug(f"Uploaded files: {', '.join([file.title for file in prompt.get_upload_files()])}")
        except Exception as e:
            # ファイルのアップロード失敗はエラーとせずに処理を続行する
            logger.warning(f"Failed to upload files: {e}")

        try:
            for file in prompt.get_upload_files():
                for _ in range(100):
                    if file.file and file.file.name and file.file.state == types.FileState.PROCESSING:
                        await asyncio.sleep(2)
                        file.file = (v := await self.client_gemini.aio.files.get(name=file.file.name))
                        logger.debug(f"Get file state: {v.state}")
        except Exception as e:
            # ファイルのアップロード失敗はエラーとせずに処理を続行する
            logger.warning(f"Failed to upload files: {e}")

        for _ in range(10):
            try:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"request({model.model.name}/{model.model_name})={json.dumps(prompt.to_gemini_rest(), ensure_ascii=False)}")
                else:
                    logger.info(
                        f"request({model.model.name}/{model.model_name})(shrink)="
                        + f"{json.dumps(shrink_message(prompt.to_gemini_rest()), ensure_ascii=False)}"
                    )
                logger.debug("Generating content")
                response = await self.client_gemini.aio.models.generate_content(
                    model=model.model_name,
                    config=prompt.to_gemini_config(),
                    contents=prompt.to_gemini_contents(),
                )
                logger.debug("Generated content")
                break
            except Exception as e:
                # しばらく待ってからリトライする
                # choices>=2 の場合にエラーが発生するケースがあるのでリトライは choices=1 とする
                prompt.choices = 1
                logger.error(e, exc_info=True)
                last_exception = e
                await asyncio.sleep(30)

        try:
            if len([True for file in prompt.get_upload_files() if file.file and file.file.name]) > 0:
                # アップロードしたファイルを削除する
                logger.debug(f"Deleting files: {', '.join([file.title for file in prompt.get_upload_files() if file.file and file.file.name])}")
                await async_utils.asyncio_gather(
                    *[
                        self.client_gemini.aio.files.delete(name=file.file.name)
                        for file in prompt.get_upload_files()
                        if file.file and file.file.name
                    ],
                    concurrency=5,
                )
                logger.debug(f"Deleted files: {', '.join([file.title for file in prompt.get_upload_files() if file.file and file.file.name])}")
                for file in prompt.get_upload_files():
                    if file.file and file.file.name:
                        file.file = None
        except Exception as e:
            # ファイルの削除失敗はエラーとせずに処理を続行する
            logger.warning(f"Failed to delete files: {e}")

        if response is None:
            # 実行に失敗していた場合は例外を送出
            raise last_exception or RuntimeError()
        logger.info(f"elapsed: {time.time() - start_time}")
        logger.info("response=" + json.dumps(response.model_dump(exclude_unset=True, exclude_defaults=True), ensure_ascii=False))
        # 利用したトークン数を加算する
        if response.usage_metadata:
            if isinstance(response.usage_metadata.prompt_token_count, int):
                model.tokens_in += response.usage_metadata.prompt_token_count
            if isinstance(response.usage_metadata.tool_use_prompt_token_count, int):
                model.tokens_in += response.usage_metadata.tool_use_prompt_token_count
            if isinstance(response.usage_metadata.candidates_token_count, int):
                model.tokens_out += response.usage_metadata.candidates_token_count
        if not response.candidates:
            # 応答がない場合は例外を送出
            logger.error("Error empty choices, response=" + str(response))
            raise last_exception or RuntimeError()
        return AiResponse.from_gemini_response(response, prompt.is_json)

    async def _chat_completions_openai(
            self,
            model: AiModelUsage,
            prompt: AiPrompt,
    ) -> AiResponse:
        if model.ai_service_type == AiServiceType.OPENAI:
            client = self.client_openai
        elif model.ai_service_type == AiServiceType.AZURE_OPENAI:
            client = self.client_azure_openai
        else:
            raise NotImplementedError(f"Unknown AiServiceType: {model.ai_service_type}")
        messages: Iterable = prompt.to_openai_messages()
        response_format = prompt.to_openai_response_format()
        tools = prompt.to_openai_tools()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"request({model.model.name}/{model.model_name})={
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
            logger.info(f"request({model.model.name}/{model.model_name})(shrink)={json.dumps(shrink_message(messages), ensure_ascii=False)}")
        start_time = time.time()
        last_exception: Optional[Exception] = None
        response = None
        for _ in range(10):
            try:
                response = await client.chat.completions.create(
                    model=model.model_name,
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
                await asyncio.sleep(30)
            except InternalServerError as e:
                if e.message == "no healthy upstream":
                    # API の一時的なエラーの場合はわずかに待ってからリトライする
                    logger.info(e)
                    last_exception = e
                    await asyncio.sleep(2)
                    continue
                logger.error(e, exc_info=True)
                raise e
            except Exception as e:
                logger.error(e, exc_info=True)
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

    async def request_classify(self, model: AiModelUsage, prompt: AiPrompt) -> dict[str, str | list[str]]:
        str_columns = ("user_intent", "user_intentions_type", "who_to_talk_to", "user_emotions",)
        list_columns = ("required_knowledge_types", "slack_emoji_names",)
        prompt.choices = 5
        response = await self._chat_completions(model, prompt)
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

    async def request_image_description(self, model: AiModelUsage, prompt: AiPrompt) -> dict[str, list[dict[str, str]]]:
        response = await self._chat_completions(model, prompt)
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

    async def request_slack_search_words(self, model: AiModelUsage, prompt: AiPrompt) -> tuple[list[str], list[str]]:
        # Gemini は n=2 で十分なバリエーションを生成してくれる
        prompt.choices = 2 if model.ai_service_type == AiServiceType.GEMINI else 5
        response = await self._chat_completions(model, prompt)
        slack_search_words = list(set(
            itertools.chain.from_iterable(
                [
                    [v for v in slack_search_words if isinstance(v, str)]
                    for message in response.choices
                    if isinstance(message.content, dict) and isinstance((slack_search_words := message.content.get('slack_search_words')), list)
                ]
            )
        ))
        external_urls = list(set(
            itertools.chain.from_iterable(
                [
                    [v for v in external_urls if isinstance(v, str)]
                    for message in response.choices
                    if isinstance(message.content, dict) and isinstance((external_urls := message.content.get('external_urls')), list)
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
        return (slack_search_words, external_urls)

    @staticmethod
    def _analyze_refine_slack_searches_response(response: AiResponse) -> list[RefineResponse]:
        return [
            RefineResponse(
                permalinks=[v for v in message.content.get('permalinks', []) if isinstance(v, str)],
                additional_search_words=[v for v in message.content.get('additional_search_words', []) if isinstance(v, str)],
            )
            for message in response.choices if (
                isinstance(message.content, dict)
                and isinstance(message.content.get('permalinks', []), list)
                and isinstance(message.content.get('additional_search_words', []), list)
            )
        ]

    async def request_refine_slack_searches(self, model: AiModelUsage, prompt: AiPrompt) -> list[RefineResponse]:
        # Gemini は n=2 で十分な網羅性がある
        prompt.choices = 2 if model.ai_service_type == AiServiceType.GEMINI else 5
        response = await self._chat_completions(model, prompt)
        return AiService._analyze_refine_slack_searches_response(response)

    async def request_url_context(self, model: AiModelUsage, prompt: AiPrompt) -> str:
        prompt.tools_url_context = True
        response = await self._chat_completions(model, prompt)
        if len(response.choices) == 0:
            return ""
        if not isinstance((v := response.choices[0].content), str):
            return ""
        return v

    @staticmethod
    def _analyze_lastshot_response(response: AiResponse) -> list[ossans_navi_type.LastshotResponse]:
        return [
            ossans_navi_type.LastshotResponse(message.content, message.images)
            for message in response.choices if isinstance(message.content, str)
        ]

    async def request_lastshot(self, model: AiModelUsage, prompt: AiPrompt) -> list[ossans_navi_type.LastshotResponse]:
        for _ in range(2):
            response = await self._chat_completions(model, prompt)
            if len(result := AiService._analyze_lastshot_response(response)) > 0:
                return result
        logger.error("Error empty choices, response=" + str(response))
        raise ValueError("Error empty choices, response=" + str(response))

    @staticmethod
    def _analyze_quality_check_response(response: AiResponse) -> Optional[QualityCheckResponse]:
        result = [
            QualityCheckResponse(
                user_intent=message.content['user_intent'],
                response_quality=message.content['response_quality'],
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

    async def request_quality_check(self, model: AiModelUsage, prompt: AiPrompt) -> QualityCheckResponse:
        for _ in range(2):
            response = await self._chat_completions(model, prompt)
            if (result := AiService._analyze_quality_check_response(response)):
                return result
        logger.error("Error empty choices, response=" + str(response))
        raise ValueError("Error empty choices, response=" + str(response))
