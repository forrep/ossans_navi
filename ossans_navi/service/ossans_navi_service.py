import datetime
import itertools
import json
import logging
import re
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Optional, overload

import html2text
from google.genai.types import Schema, Type
from PIL import Image, ImageFile

from ossans_navi import config
from ossans_navi.common.cache import LRUCache
from ossans_navi.service.ai_prompt_service import AiPromptService
from ossans_navi.service.ai_service import (AiModels, AiPrompt, AiPromptContent, AiPromptImage, AiPromptMessage, AiPromptRole, AiService,
                                            QualityCheckResponse)
from ossans_navi.service.slack_service import SlackService
from ossans_navi.type.ossans_navi_types import OssansNaviConfig
from ossans_navi.type.slack_type import SlackFile, SlackMessage, SlackMessageEvent, SlackMessageLite, SlackSearches, SlackSearchTerm

logger = logging.getLogger(__name__)


class OssansNaviService:
    image_cache = LRUCache[str, list[dict[str, str]]](capacity=500, expire=1 * 60 * 60 * 24)

    def __init__(self, ai_service: AiService, slack_service: SlackService, models: AiModels, event: SlackMessageEvent) -> None:
        self.ai_service = ai_service
        self.slack_service = slack_service
        self.ai_prompt_service = AiPromptService(event, self.slack_service.get_assistant_names())
        self.models = models
        self.event = event
        self.config = self.get_config()
        self.slack_searches = SlackSearches()
        self.slack_file_permalinks: dict[str, SlackFile] = {}
        """同一 permalink に紐づく SlackFile を一つのインスタンスにまとめるために permalink から SlackFile を取得する辞書"""

    @staticmethod
    def json_dumps_converter(v) -> str:
        if isinstance(v, datetime.datetime):
            return v.strftime('%Y-%m-%d %H:%M:%S')
        raise TypeError(f"Object of type {type(v).__name__} is not JSON serializable")

    @overload
    @staticmethod
    def slack_message_to_ai_prompt(
        message: SlackMessage | SlackMessageLite,
        limit: int = 800,
        ellipsis: str = "...",
        check_dup_files: bool = False,
        check_dup_files_dict: Optional[dict[str, int]] = None,
        allow_private_files: bool = False,
    ) -> dict[str, Any]:
        ...

    @overload
    @staticmethod
    def slack_message_to_ai_prompt(
        message: list[SlackMessage],
        limit: int = 800,
        ellipsis: str = "...",
        check_dup_files: bool = False,
        check_dup_files_dict: Optional[dict[str, int]] = None,
        allow_private_files: bool = False,
    ) -> list[dict[str, Any]]:
        ...

    @staticmethod
    def slack_message_to_ai_prompt(
        message: SlackMessage | SlackMessageLite | list[SlackMessage],
        limit: int = 800,
        ellipsis: str = "...",
        check_dup_files: bool = False,
        check_dup_files_dict: Optional[dict[str, int]] = None,
        allow_private_files: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        check_dup_files_dict = {} if check_dup_files_dict is None else check_dup_files_dict
        if isinstance(message, list):
            return [
                OssansNaviService.slack_message_to_ai_prompt(
                    v,
                    limit,
                    ellipsis,
                    check_dup_files,
                    check_dup_files_dict,
                ) for v in message
            ]
        if isinstance(message, SlackMessage):
            return {
                **(
                    OssansNaviService.slack_message_to_ai_prompt(
                        message.message,
                        limit,
                        ellipsis,
                        check_dup_files,
                        check_dup_files_dict,
                    )
                ),
                "channel": message.channel,
                **(
                    {
                        "root_message": OssansNaviService.slack_message_to_ai_prompt(
                            v,
                            limit,
                            ellipsis,
                            check_dup_files,
                            check_dup_files_dict,
                        )
                    } if (v := message.root_message) else {}
                ),
                **(
                    {
                        "replies": [OssansNaviService.slack_message_to_ai_prompt(
                            v,
                            limit,
                            ellipsis,
                            check_dup_files,
                            check_dup_files_dict,
                        ) for v in message.messages]
                    } if len(message.messages) > 0 else {}
                ),
            }
        if isinstance(message, SlackMessageLite):
            return {
                "timestamp": message.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "name": message.user.name,
                "user_id": message.user.mention,
                "content": message.content[:limit] + (ellipsis if len(message.content) > limit else ""),
                "permalink": message.permalink,
                **(
                    {
                        "attachments": [
                            {
                                "title": v.title[:limit] + (ellipsis if len(v.title) > limit else ""),
                                "text": v.text[:limit] + (ellipsis if len(v.text) > limit else ""),
                                "link": v.link,
                                **(
                                    {
                                        "name": v.user.name,
                                        "user_id": v.user.user_id
                                    } if v.user else {}
                                )
                            } for v in message.attachments
                        ]
                    } if len(message.attachments) > 0 else {}
                ),
                **(
                    {
                        "files": [
                            {
                                "title": v.title,
                                "link": v.permalink,
                                **(
                                    {} if (
                                        # 重複ファイルをチェックする、かつ1度登場している場合は、description と text を入力しない
                                        check_dup_files
                                        and (1 if v.permalink in check_dup_files_dict else check_dup_files_dict.setdefault(v.permalink, 0)) > 0
                                    ) else {
                                        **(
                                            {
                                                "description": v.description[:limit] + (ellipsis if len(v.description) > limit else "")
                                            } if v.description else {}
                                        ),
                                        **(
                                            {
                                                "text": v.text[:limit] + (ellipsis if len(v.text) > limit else "")
                                            } if v.text else {}
                                        ),
                                    }
                                )
                            } for v in message.files if v.is_textualize and (allow_private_files or v.is_public)
                        ]
                    } if len([v for v in message.files if v.is_textualize and (allow_private_files or v.is_public)]) > 0 else {}
                ),
                **(
                    {
                        "reactions": [reaction for reaction in message.reactions],
                    } if len(message.reactions) > 0 else {}
                ),
            }

    def get_ai_prompt(
        self,
        system: str,
        messages: list[SlackMessageLite],
        with_image_files: bool = False,
        input_image_files: int = 0,
        limit: int = 800,
        limit_last_message: int = -1,
        schema: Optional[Schema] = None,
        is_json: bool = True,
    ) -> AiPrompt:
        if limit_last_message < 0:
            limit_last_message = limit

        return AiPrompt(
            system=system,
            messages=[
                AiPromptMessage(
                    role=AiPromptRole.ASSISTANT if message.user.user_id in self.slack_service.my_bot_user_id else AiPromptRole.USER,
                    content=AiPromptContent(
                        data=OssansNaviService.slack_message_to_ai_prompt(
                            message,
                            limit=(limit_last_message if i == 0 else limit),
                            allow_private_files=True,
                        ),
                        images=(
                            [
                                AiPromptImage(data=file.content)
                                for file in message.files if file.is_image and not file.is_analyzed and file.is_valid
                            ]
                            if with_image_files and message.has_not_analyzed_files() else (
                                [
                                    # input_image_files に指定した枚数だけ画像ファイルを再度入力する
                                    # lastshot で画像そのものを入力した方が精度が上がるため、安価な Gemini 限定で入力する
                                    # また今後画像出力に対応する場合に、元画像を編集する用途には元画像そのものを入力する必要がある
                                    AiPromptImage(data=file.content)
                                    for file in message.files
                                    if (
                                        file.is_image
                                        and file.is_valid
                                        and (input_image_files := input_image_files - 1) >= 0
                                        and input_image_files >= 0  # 本来この条件は必要ないが Flake8 が input_image_files を利用していないと誤判定する対策
                                    )
                                ]
                                # input_image_files > 0 、かつ AiPromptRole.USER の場合だけ画像を入力する
                                # 現時点では AiPromptRole.ASSISTANT で画像を入力すると以下の動作になる
                                #   OpenAI → 画像を入力するとAPIエラーが発生する
                                #   Gemini → 画像を入力しても参照されない（APIエラーは発生しない）
                                if input_image_files > 0 and message.user.user_id not in self.slack_service.my_bot_user_id
                                else []
                            )
                        )

                    ),
                    name=message.user.user_id,
                )
                # 最後のメッセージから逆順に処理する（[::-1]）
                # なぜなら枚数限定で画像入力する際に最新メッセージを優先するため
                # 生成後に再度逆順にすることで元の順番に戻す
                for (i, message) in enumerate(messages[::-1])
            ][::-1],
            schema=schema,
            is_json=is_json,
        )

    @overload
    def _integrate_duplicated_slack_file(self, messages: SlackMessageLite) -> SlackMessageLite:
        ...

    @overload
    def _integrate_duplicated_slack_file(self, messages: list[SlackMessageLite]) -> list[SlackMessageLite]:
        ...

    def _integrate_duplicated_slack_file(self, messages: SlackMessageLite | list[SlackMessageLite]) -> SlackMessageLite | list[SlackMessageLite]:
        if isinstance(messages, SlackMessageLite):
            messages.files = [self.slack_file_permalinks.setdefault(file.permalink, file) for file in messages.files]
            return messages
        elif isinstance(messages, list):
            for message in messages:
                message.files = [self.slack_file_permalinks.setdefault(file.permalink, file) for file in message.files]
            return messages
        raise TypeError(f"invalid type {type(messages)}")

    def get_thread_messages(self) -> list[SlackMessageLite]:
        # SlackMessageLite を取得した後に _integrate_duplicated_slack_file で含まれる SlackFile のインスタンスを1つにまとめる
        thread_messages = self._integrate_duplicated_slack_file(
            self.slack_service.get_conversations_replies(self.event.channel_id, self.event.thread_ts)
        )
        logger.info(
            "conversations_replies="
            + json.dumps([v.to_dict() for v in thread_messages], ensure_ascii=False, default=OssansNaviService.json_dumps_converter)
        )

        # スレッドのコンテキスト制限、3メッセージ以上ある場合が削減対象
        while (
            len(thread_messages) >= 3
            and (
                self.models.low_cost.tokenizer.messages_tokens(
                    self.get_ai_prompt("", thread_messages).to_openai_prompt()
                ) > config.MAX_THREAD_TOKENS
            )
        ):
            if len((replies := [message for message in thread_messages[1:] if len(message.files) + len(message.attachments) > 0])) > 0:
                # スレッド内のファイルとアタッチメントを最もメッセージから削除する
                replies[0].files = []
                replies[0].attachments = []
                continue
            if len(thread_messages[0].files) + len(thread_messages[0].attachments) > 0:
                # 親メッセージのファイルとアタッチメントを削除する
                thread_messages[0].files = []
                thread_messages[0].attachments = []
                continue
            # 古い順に1メッセージ削除する [1, 2, 3, 4, 5] → [1, 3, 4, 5]
            thread_messages = [thread_messages[0], *thread_messages[2:]]
        logger.info(
            "conversations_replies(omit)="
            + json.dumps([v.to_dict() for v in thread_messages], ensure_ascii=False, default=OssansNaviService.json_dumps_converter)
        )
        return thread_messages

    def is_joined(self, thread_messages: list[SlackMessageLite]) -> bool:
        return len([reply for reply in thread_messages if reply.user.user_id in self.slack_service.my_bot_user_id]) > 0

    def is_next_message_from_ossans_navi(self, thread_messages: list[SlackMessageLite]) -> bool:
        return len(thread_messages) >= 2 and thread_messages[-2].user.user_id in self.slack_service.my_bot_user_id

    def classify(self, thread_messages: list[SlackMessageLite]) -> dict[str, str | list[str]]:
        return self.ai_service.request_classify(
            self.models.low_cost,
            self.get_ai_prompt(
                self.ai_prompt_service.classify_prompt(),
                thread_messages,
                schema=self.ai_prompt_service.CLASSIFY_SCHEMA,
            )
        )

    def store_config(self, config: OssansNaviConfig) -> None:
        self.slack_service.store_config_dict(config.to_dict())

    def get_config(self) -> OssansNaviConfig:
        config_dict = self.slack_service.get_config_dict()
        return OssansNaviConfig.from_dict(config_dict)

    def special_command(self) -> bool:
        """
        管理者が ossans_navi との DM に `config trusted_bots show` などのメッセージを送信することで利用できる special_command の実行判定及び実行を行う
        special_command では OssansNavi の設定を行うことができる
        special_command を実行した場合は True を返す、special_command に該当しなければ False を返すので、呼び出し元は通常のメッセージとして扱う
        呼び出し元は False が返った場合は元の処理を継続する、True の場合は special_command を実行しているので処理を打ち切る
        """
        if (
            match := re.match(
                r'config\s+(trusted_bots|allow_responds|admin_users|viewable_private_channels)\s+(show|add|remove)(?:\s+(.+))?',
                self.event.text
            )
        ):
            if (
                not (
                    # 管理者権限を持つユーザーのみ special_command を実行可能、管理者権限のルールは以下のいずれかを満たすこと
                    #   - Slackワークスペースの admin 権限を持っていること
                    #   - ossans_navi 設定の admin_users が空で、かつ ossans_navi アプリをインストールしたユーザーであること
                    #   - ossans_navi 設定の admin_users に含まれていること
                    self.event.user.is_admin
                    or (len(self.config.admin_users) == 0 and self.event.user.user_id == self.slack_service.my_user_id)
                    or self.event.user.user_id in self.config.admin_users
                )
            ):
                # 管理者権限を持っていないので special_command を実行しないで終了
                logger.info(f"permission denied. cannot execute special_command. event={self.event}")
                return False
            category: str = match.group(1)
            action: str = match.group(2)
            value: Optional[str] = match.group(3)
            value = (value.strip() if isinstance(value, str) else None)
            text = f"{category} {action}{f" {value}" if value else ""}"
            target_user = None
            if (
                category in ("trusted_bots", "allow_responds", "admin_users")
                and action in ("add", "remove")
                and value
            ):
                value = re.sub(r'<@([A-Z0-9]+)(\|[^>]+?)?>', "\\1", value)
                if (v := self.slack_service.get_user(value)).is_valid:
                    target_user = v

            if category == "trusted_bots":
                # 現在有効な trusted_bots を取得
                now_users = {
                    **(
                        {
                            user.user_id: user
                            for user in [self.slack_service.get_user(user_id) for user_id in self.config.trusted_bots]
                            if user.is_valid
                        }
                    ),
                }
                if action in ("add", "remove") and value:
                    if target_user is not None and target_user.is_bot:
                        if action == "add":
                            now_users[target_user.user_id] = target_user
                            text += f"\nadded: <@{target_user.user_id}>"
                            logger.info(f"{category} added: {target_user.name}({target_user.user_id})")
                        if action == "remove" and target_user.user_id in now_users:
                            del now_users[target_user.user_id]
                            text += f"\nremoved: <@{target_user.user_id}>"
                            logger.info(f"{category} removed: {target_user.name}({target_user.user_id})")
                        self.config.trusted_bots = list(now_users.keys())
                        self.slack_service.store_config_dict(self.config.to_dict())
                text += ("\n• " if len(now_users) > 0 else "\nempty")
                text += "\n• ".join([f"<@{bot.user_id}>" for bot in now_users.values()])
                self.slack_service.chat_post_message(channel=self.event.channel_id, thread_ts=self.event.thread_ts, text=text)
            if category == "allow_responds":
                # 現在有効な allow_responds を取得
                now_users = {
                    **(
                        {
                            user.user_id: user
                            for user in [self.slack_service.get_user(user_id) for user_id in self.config.allow_responds]
                            if user.is_valid
                        }
                    ),
                }
                if action in ("add", "remove") and value:
                    if target_user is not None:
                        if action == "add":
                            now_users[target_user.user_id] = target_user
                            text += f"\nadded: <@{target_user.user_id}>"
                            logger.info(f"{category} added: {target_user.name}({target_user.user_id})")
                        if action == "remove" and target_user.user_id in now_users:
                            del now_users[target_user.user_id]
                            text += f"\nremoved: <@{target_user.user_id}>"
                            logger.info(f"{category} removed: {target_user.name}({target_user.user_id})")
                        self.config.allow_responds = list(now_users.keys())
                        self.slack_service.store_config_dict(self.config.to_dict())
                text += ("\n• " if len(now_users) > 0 else "\nempty")
                text += "\n• ".join([f"<@{bot.user_id}>" for bot in now_users.values()])
                self.slack_service.chat_post_message(channel=self.event.channel_id, thread_ts=self.event.thread_ts, text=text)
            if category == "admin_users":
                # 現在有効な admin_users を取得
                now_users = {
                    user.user_id: user
                    for user in [self.slack_service.get_user(user_id) for user_id in self.config.admin_users]
                    if user.is_valid
                }
                if action in ("add", "remove") and value:
                    if target_user is not None and not target_user.is_bot and not target_user.is_guest:
                        if action == "add":
                            now_users[target_user.user_id] = target_user
                            text += f"\nadded: <@{target_user.user_id}>"
                            logger.info(f"{category} added: {target_user.name}({target_user.user_id})")
                        if action == "remove" and target_user.user_id in now_users:
                            del now_users[target_user.user_id]
                            text += f"\nremoved: <@{target_user.user_id}>"
                            logger.info(f"{category} removed: {target_user.name}({target_user.user_id})")
                        self.config.admin_users = list(now_users.keys())
                        self.slack_service.store_config_dict(self.config.to_dict())
                text += ("\n• " if len(now_users) > 0 else "\nempty")
                text += "\n• ".join([f"<@{user.user_id}>" for user in now_users.values()])
                self.slack_service.chat_post_message(channel=self.event.channel_id, thread_ts=self.event.thread_ts, text=text)
            if category == "viewable_private_channels":
                #  現在有効な viewable_private_channels を取得
                now_channels = {
                    channel.channel_id: channel
                    for channel in [
                        self.slack_service.get_channel(channel_id, True) for channel_id in self.config.viewable_private_channels
                    ]
                    if channel.is_valid and channel.is_private
                }
                if action in ("add", "remove") and value:
                    value = re.sub(r'<#([A-Z0-9]+)(\|[^>]*?)?>', "\\1", value)
                    target_channel = self.slack_service.get_channel(value, True)
                    if target_channel.is_valid and target_channel.is_private:
                        if action == "add":
                            now_channels[target_channel.channel_id] = target_channel
                            text += f"\nadded: <#{target_channel.channel_id}>"
                            logger.info(f"{category} added: {target_channel.name}({target_channel.channel_id})")
                        if action == "remove" and target_channel.channel_id in now_channels:
                            del now_channels[target_channel.channel_id]
                            text += f"\nremoved: <#{target_channel.channel_id}>"
                            logger.info(f"{category} removed: {target_channel.name}({target_channel.channel_id})")
                        self.config.viewable_private_channels = list(now_channels.keys())
                        self.slack_service.store_config_dict(self.config.to_dict())
                text += ("\n• " if len(now_channels) > 0 else "\nempty")
                text += "\n• ".join([f"<#{channel.channel_id}>" for channel in now_channels.values()])
                self.slack_service.chat_post_message(channel=self.event.channel_id, thread_ts=self.event.thread_ts, text=text)
            return True
        return False

    @staticmethod
    def load_image_description_from_cache(thread_messages: list[SlackMessageLite]):
        for message in thread_messages:
            if message.has_files() and len([file for file in message.files if file.is_image and not file.is_analyzed]) > 0:
                if (cached := OssansNaviService.image_cache.get(message.permalink)).found:
                    for (file, cached_file) in itertools.zip_longest(message.files, cached.value):
                        if file is None:
                            break
                        file.is_analyzed = True
                        file.description = cached_file.get("description", "") if cached_file is not None else ""
                        file.text = cached_file.get("text", "") if cached_file is not None else ""
                else:
                    for file in message.files:
                        file.is_analyzed = False

    def analyze_image_description(self, thread_messages: list[SlackMessageLite]):
        # キャッシュから読み込めるやつは読み込んでおく、ここで読み込まれた分は生成AIに渡されないからトークンの節約になる
        OssansNaviService.load_image_description_from_cache(thread_messages)
        messages_token = self.models.high_quality.tokenizer.messages_tokens(
            self.get_ai_prompt(
                self.ai_prompt_service.image_description_prompt(),
                thread_messages,
            ).to_openai_prompt()
        )
        for i in range(len(thread_messages)):
            if (
                messages_token
                + sum(
                    [
                        self.models.high_quality.tokenizer.image_tokens(file.width, file.height)
                        for file in itertools.chain.from_iterable(
                            [message.files for message in thread_messages]
                        ) if file.is_image and not file.is_analyzed
                    ]
                )
            ) < 20000:
                # メッセージ+画像のトークン数が20000を下回るならばそのままAI入力する
                break
            else:
                # 20000トークンを上回る場合は古いメッセージから file を空データで解析済みとする（つまりAI入力から除外する）
                for file in thread_messages[i].files:
                    file.is_analyzed = True
                    file.description = ""
                    file.text = ""
        if (
            len(
                [
                    1 for file in itertools.chain.from_iterable([message.files for message in thread_messages])
                    if file.is_image and not file.is_analyzed
                ]
            ) == 0
        ):
            # 画像かつ未分析（not is_analyzed）のファイルが1枚もないなら終了、分析対象がない
            return

        # 添付画像を AI で解析実行
        image_description = self.ai_service.request_image_description(
            self.models.high_quality,
            self.get_ai_prompt(
                self.ai_prompt_service.image_description_prompt(),
                thread_messages,
                with_image_files=True,
                schema=Schema(
                    type=Type.OBJECT,
                    properties={
                        message.permalink: self.ai_prompt_service.IMAGE_DESCRIPTION_SCHEMA for message in thread_messages
                    },
                    required=[message.permalink for message in thread_messages]
                )
            )
        )
        logger.info("image_description=" + json.dumps(image_description, ensure_ascii=False))
        # 画像の解釈結果をキャッシュに積む
        for (permalink, analyzed) in image_description.items():
            OssansNaviService.image_cache.put(permalink, analyzed)
        # キャッシュに追加した後で再度キャッシュから情報を読み込む
        OssansNaviService.load_image_description_from_cache(thread_messages)

    def search(
        self,
        terms_strs: list[str],
        is_additional: bool = False,
        is_get_messages: bool = False
    ) -> None:
        # 文字列の検索ワードをパースして SlackSearchTerm に変換、同時に無効な条件を削除
        terms = [term for term in [SlackSearchTerm.parse(term_str) for term_str in terms_strs] if term is not None]

        # 絞り込みのキーワード数が少なく、文字数も少ない検索条件から順番に検索する
        # なぜならば「ワードA AND ワードB」の検索結果が 10件ならば 「ワードA AND ワードB AND ワードC」は実行する必要がなくスキップできるため
        for current_term in sorted(terms):
            if (
                any(
                    [
                        (result.is_full and result.term.is_subset(current_term)) or result.term == current_term
                        for result in self.slack_searches.results
                    ]
                )
            ):
                # 今回の検索条件（current_term）が部分集合（サブセット）となる検索済み結果が slack_searches 内にすでにあり
                # かつ、それが全件取得済み（is_full）ならば current_term は検索の必要が無いのでスキップする
                # つまり「ワードA AND ワードB」の検索結果を 10件取得済みならば 「ワードA AND ワードB AND ワードC」を検索する必要が無い
                continue

            result = self.slack_service.search(
                current_term,
                self.event.user,
                self.event.channel_id,
                self.event.thread_ts,
                self.config.viewable_private_channels,
                self.config.trusted_bots,
                is_additional,
                is_get_messages,
            )
            # 複数のメッセージに同一のファイルが紐づく可能性がある。
            # それらは別のインスタンスとなっているので SlackFile.permalink 単位で一つのインスタンスにまとめる
            self._integrate_duplicated_slack_file([message.message for message in result.messages])
            self.slack_searches.add(result)
            if result.term.date_from is None and result.term.date_to is None:
                if result.is_meny_messages():
                    self.slack_searches.add(SlackService.duplicate_search_result(result, 2))
                if result.is_too_meny_messages():
                    self.slack_searches.add(SlackService.duplicate_search_result(result, 1))

    def do_slack_searches(self, thread_messages: list[SlackMessageLite]) -> Generator[None, None, None]:
        # Slack ワークスペースを検索するワードを生成してもらう
        # function calling を利用しない理由は、適切にキーワードを生成してくれないケースがあったり、実行回数などコントロールしづらいため
        # 具体的には \n や空白文字が連続したり、「あああああ」みたいな意味不明なキーワードが生成されたり、指示しても1つのキーワードしか生成してくれないケースなど
        # 普通にレスポンスとしてキーワードを生成してもらった方が高い精度で生成される
        request_slack_search_words_prompt = self.get_ai_prompt(
            self.ai_prompt_service.slack_search_word_prompt(),
            thread_messages,
            schema=self.ai_prompt_service.SLACK_SEARCH_WORD_SCHEMA,
        )

        for i in range(3):
            # 呼び出し元にイベントのキャンセル確認させるために定期的に yield で処理を戻す
            yield

            slack_search_words = self.ai_service.request_slack_search_words(self.models.high_quality, request_slack_search_words_prompt)

            # Slack API でキーワード検索を実行して self.slack_searches に積み込む処理
            # slack_searches 内ではヒット件数（total_count）が少ない方がより絞り込めた良いキーワードと判断してヒット件数の昇順で並べる
            self.search(slack_search_words)

            if self.slack_searches.result_len() == 0:
                # キーワード自体が生成されなかったケース、つまり検索が必要がない質問やメッセージに対する応答
                # Slack検索フェーズを終了する
                break
            if i == 0 and self.slack_searches.total_count >= 10:
                # 最初の検索(i==0)、かつ検索結果が10件以上ヒットした場合は Slack検索フェーズを終了
                break
            if i > 0 and self.slack_searches.total_count >= 1:
                # 2回目以降の検索(i>0)、かつ検索結果が1件以上ヒットした場合は Slack検索フェーズを終了
                # 2回検索しても1件しかヒットしないなら諦める
                break
            # ここまで到達したらまだ別の検索ワードが必要という判断
            # もう一周、AIにキーワードを生成してもらう（AIにキーワードを生成してもらうフェーズは全体で3回まで）
            for slack_search in self.slack_searches:
                logger.info(f"total_count={slack_search.total_count}, words={slack_search.words}, id={slack_search.get_id()}")
            request_slack_search_words_prompt.messages.append(
                AiPromptMessage(
                    role=AiPromptRole.USER,
                    content=AiPromptContent(
                        data=(
                            f"以下の検索ワードで検索してみましたが、{"結果がヒットしませんでした" if self.slack_searches.total_count == 0 else "ヒット件数が少なかったです"}。"
                            + "出力フォーマットに従って別のSlack検索キーワードを提供してください。\n"
                            + "検索キーワードを別の表現にしたり、AND検索による絞り込みをやめて1単語だけするなど工夫してください\n"
                            + "\n"
                            + "## 検索キーワードとヒット件数\n"
                            + "\n".join([f"検索ワード: {v.words}, ヒット件数: {len(v.messages)}件" for v in self.slack_searches])
                        )
                    ),
                )
            )
        yield

    def refine_slack_searches(self, thread_messages: list[SlackMessageLite]) -> Generator[None, None, None]:
        """
        slack_searches の結果から有用な情報を抽出するフェーズ（refine_slack_searches）
        トークン数の上限があるので複数回に分けて実行して、大量の検索結果の中から必要な情報を絞り込む
        RAG で入力する情報以外のトークン数を求めておく（システムプロンプトなど）、RAG で入力可能な情報を計算する為に使う
        """
        base_messages_token = self.models.low_cost.tokenizer.messages_tokens(
            self.get_ai_prompt(
                self.ai_prompt_service.refine_slack_searches_prompt([], [v.words for v in self.slack_searches]),
                thread_messages,
            ).to_openai_prompt()
        )
        logger.info(f"{base_messages_token=}")
        # メンションされた場合か、OssansNavi のメッセージの次のメッセージの場合はちゃんと調べる、それ以外は手を抜いて調べる
        if self.event.is_mention or self.event.is_reply_to_ossans_navi():
            refine_slack_searches_count = config.REFINE_SLACK_SEARCHES_COUNT_WITH_MENTION = 4
            refine_slack_searches_depth = config.REFINE_SLACK_SEARCHES_DEPTH_WITH_MENTION
        else:
            refine_slack_searches_count = config.REFINE_SLACK_SEARCHES_COUNT_NO_MENTION
            refine_slack_searches_depth = config.REFINE_SLACK_SEARCHES_DEPTH_NO_MENTION
        for i in range(refine_slack_searches_depth):
            with ThreadPoolExecutor(max_workers=config.REFINE_SLACK_SEARCHES_THREADS, thread_name_prefix="RefineWorker") as executor:
                # 最後の refine かどうか？最後以外は新たな検索ワードを追加する処理などがある
                is_last_refine = i + 1 == 2
                for _ in range(refine_slack_searches_count):
                    executor.submit(self._refine_slack_searches_safe, thread_messages, base_messages_token, is_last_refine)
            yield

    def _refine_slack_searches_safe(
        self,
        thread_messages: list[SlackMessageLite],
        base_messages_token: int,
        is_last_refine: bool
    ) -> None:
        try:
            self._refine_slack_searches(thread_messages, base_messages_token, is_last_refine)
        except Exception as e:
            logger.error(e, exc_info=True)

    def _refine_slack_searches(
        self,
        thread_messages: list[SlackMessageLite],
        base_messages_token: int,
        is_last_refine: bool
    ) -> None:
        """
        slack_searches の結果から有用な情報を抽出するフェーズを非同期で行う
        """
        # slack_searches から入力候補を抽出する処理は同期的に行う（slack_searches でロックを取得）
        with self.slack_searches:
            # 入力可能な残りトークン数を保持する、0未満にならないように管理する
            tokens_remain = config.REFINE_SLACK_SEARCHES_TOKEN - base_messages_token
            tokens_full: bool = False
            current_messages: list[SlackMessage] = []
            for slack_search in self.slack_searches:
                if tokens_full:
                    # 入力可能な余剰がなくなったら終了する
                    break
                # slack_search は優先度が高い順に並んでいるので、最初の候補から順番に利用する
                # 1回の検索ワードあたり 5000 トークン以内に収まる範囲で入力する
                # 安価な GPT-4o mini で slack_search の結果を精査して、lastshot に入力する情報を slack_searches.add_lastshot() に積むのが目的
                candidate_messages: list[SlackMessage] = []
                for message in slack_search.messages:
                    # slack_searches のロックを取得してから、このメッセージがすでに使われているか？などの判定を行って、refine 対象の message を決定する
                    if self.slack_searches.is_used(message.permalink):
                        # ヒットしたメッセージがすでにAI入力済みだった場合は次へ
                        continue
                    # スレッド情報などを取得する（Slack API実行のため多少時間がかかる）
                    self.load_slack_message(message)
                    # 今回の検索結果を追加した場合のトークン数を試算する
                    tokens = self.models.low_cost.tokenizer.content_tokens(json.dumps(
                        [OssansNaviService.slack_message_to_ai_prompt(message) for message in [*candidate_messages, message]],
                        ensure_ascii=False,
                        separators=(',', ':')
                    ))
                    if tokens > tokens_remain:
                        # 試算結果が入力可能トークン数を上回るなら入力せずに終了
                        tokens_full = True
                        break
                    if len(candidate_messages) >= 1 and tokens > 5000:
                        # 1つの検索ワードに対してトークン数（試算）が 5000 未満の範囲で入力する、ただし1つのメッセージで 5000 トークンを越える場合は許容する
                        # 1つの検索ワードで大量の情報を入れるのではなく、多様性のために様々な検索ワードによる検索結果を採用したいため
                        break
                    # ここまで到達したなら残トークン数に問題がないということ、入力することが決定
                    candidate_messages.append(message)
                    # メッセージそのものと、root_message の permlink を use 状態にしておく、そして別の検索ワードで同一メッセージが何度も引っかかるのを防ぐ
                    # なぜならば、同一メッセージを何度も入力しても無駄だから
                    self.slack_searches.use(message.permalink)
                    if message.is_full:
                        self.slack_searches.use([v.permalink for v in message.messages])
                        if message.root_message:
                            self.slack_searches.use(message.root_message.permalink)
                # 今回入力するトークン数を slack_searches_tokens_remain から引いておく
                tokens_remain -= self.models.low_cost.tokenizer.content_tokens(json.dumps(
                    [OssansNaviService.slack_message_to_ai_prompt(message) for message in candidate_messages],
                    ensure_ascii=False,
                    separators=(',', ':')
                ))
                logger.info(
                    f"words={slack_search.words}{f" (Additional)" if slack_search.is_additional else ""},"
                    + f" candidate_messages={len(candidate_messages)}, "
                    + f"content_count={len(slack_search.messages)} total_count={slack_search.total_count}, "
                    + f"tokens_remain={tokens_remain}, id={slack_search.get_id()}"
                )
                current_messages.extend(candidate_messages)
            if len(current_messages) == 0:
                # ヒット件数がゼロ件などの理由で入力可能な検索結果が存在しなかったら refine_slack_searches フェーズを終了する
                logger.info("current_messages is empty, finished.")
                return
            refine_slack_searches_prompt = self.get_ai_prompt(
                self.ai_prompt_service.refine_slack_searches_prompt(
                    [OssansNaviService.slack_message_to_ai_prompt(message) for message in SlackMessage.sort(current_messages)],
                    [v.words for v in self.slack_searches if not v.is_get_messages]
                ),
                thread_messages,
                schema=self.ai_prompt_service.REFINE_SLACK_SEARCHES_SCHEMA,
            )

        # AI への問い合わせ部分だけ並列で処理する
        refine_slack_searches_responses = self.ai_service.request_refine_slack_searches(
            self.models.low_cost,
            refine_slack_searches_prompt
        )
        logger.info(f"{refine_slack_searches_responses=}")

        if len(refine_slack_searches_responses) == 0:
            # 返答が空のケース、普通はないはずだけど AI はどうしても誤動作する可能性があるので、稀にこのパターンも発生する
            # その回の検索結果は諦めて次にトライ、精度は下がるけど仕方なし
            logger.error("refine_slack_searches_responses is emtpy, continue.")
            return

        # response を slack_searches へ反映する処理は同期的に行う（slack_searches でロックを取得）
        with self.slack_searches:
            # 参考になった permalink は lastshot で利用するので保存しておく
            # get_next_messages も参考になった permalink なので追加する
            for refine_slack_searches_response in refine_slack_searches_responses:
                self.slack_searches.add_lastshot(refine_slack_searches_response.permalinks)
                self.slack_searches.add_lastshot(refine_slack_searches_response.get_next_messages)
                if not is_last_refine:
                    # 最後の refine ではない→ まだ検索結果を精査するフェーズが残っている
                    if len(refine_slack_searches_response.additional_search_words) > 0:
                        # 追加の検索ワードが提供された場合は検索する
                        self.search(refine_slack_searches_response.additional_search_words, True)
                    if len(refine_slack_searches_response.get_messages) > 0:
                        # 追加の取得メッセージが提供された場合は検索する
                        self.search(refine_slack_searches_response.get_messages, True, True)

    def lastshot(self, thread_messages: list[SlackMessageLite]) -> list[str]:
        current_messages: list[SlackMessage] = []
        # 入力可能なトークン数を定義する、たくさん入れたら精度が上がるが費用も上がるのでほどほどのトークン数に制限する（話しかけられている時はトークン量を増やす）
        if self.event.is_mention or self.event.is_reply_to_ossans_navi():
            tokens_remain = config.LASTSHOT_TOKEN_WITH_MENTION
            n = 2
        else:
            tokens_remain = config.LASTSHOT_TOKEN_NO_MENTION
            n = 1
        tokens_remain -= self.models.high_quality.tokenizer.messages_tokens(
            self.get_ai_prompt(
                self.ai_prompt_service.lastshot_prompt(),
                thread_messages,
                limit=10000,
                limit_last_message=30000,
            ).to_openai_prompt()
        )
        # GPT-4o mini が精査してくれた結果を元にトークン数が収まる範囲で入力データとする
        check_dup_files_dict: dict[str, int] = {}
        for content in self.slack_searches.lastshot_messages:
            tokens = self.models.high_quality.tokenizer.content_tokens(
                json.dumps(
                    OssansNaviService.slack_message_to_ai_prompt(
                        content,
                        limit=10000,
                        check_dup_files=True,
                        check_dup_files_dict=check_dup_files_dict
                    ),
                    ensure_ascii=False,
                    separators=(',', ':')
                )
            )
            if tokens > tokens_remain:
                # 入力できるトークン数が slack_searches_tokens_remain を越えたら終了
                break
            tokens_remain -= tokens
            current_messages.append(content)
        logger.info(f"Lastshot input_count={len(current_messages)}")

        return self.ai_service.request_lastshot(
            self.models.high_quality,
            self.get_ai_prompt(
                self.ai_prompt_service.lastshot_prompt(
                    OssansNaviService.slack_message_to_ai_prompt(SlackMessage.sort(current_messages), limit=10000, check_dup_files=True)
                ),
                thread_messages,
                input_image_files=config.LASTSHOT_INPUT_IMAGE_FILES,
                limit=10000,
                limit_last_message=30000,
                is_json=False,
            ),
            n
        )

    def quality_check(self, thread_messages: list[SlackMessageLite], response_message: str) -> QualityCheckResponse:
        return self.ai_service.request_quality_check(
            self.models.low_cost,
            self.get_ai_prompt(
                self.ai_prompt_service.quality_check_prompt(response_message),
                thread_messages,
                limit=2000,
                limit_last_message=10000,
                schema=self.ai_prompt_service.QUALITY_CHECK_SCHEMA,
            ),
        )

    def load_slack_file(self, file: SlackFile, user_client: bool = False) -> None:
        if file.is_initialized:
            # ロード済みなら処理せずに終了
            return
        file.is_initialized = True
        try:
            file.content = self.slack_service.load_file(file.download_url, user_client)
            if file.is_image:
                bytes_io = BytesIO(file.content)
                image: Image.Image | ImageFile.ImageFile = Image.open(bytes_io)
                max_size = max(image.width, image.height)
                if max_size > config.MAX_IMAGE_SIZE:
                    resize_retio = config.MAX_IMAGE_SIZE / max_size
                    image = image.resize((int(image.width * resize_retio), int(image.height * resize_retio)))
                bytes_io = BytesIO()
                image.save(bytes_io, format="PNG")
                file.content = bytes_io.getvalue()
                file.mimetype = 'image/png'
                file._height = image.height
                file._width = image.width
            elif file.is_text:
                file.text = file.content.decode("utf-8")
            elif file.is_canvas:
                html_content = file.content.decode("utf-8")
                parser = html2text.HTML2Text()
                parser.images_to_alt = True
                html_parsed = parser.handle(html_content)
                file.text = html_parsed
                file.mimetype = "text/markdown"
                file.filetype = "markdown"
                file.pretty_type = "Markdown"
        except Exception as e:
            logger.info(f"Slack get_image failed: {str(e)}")

    def load_slack_message(self, message: SlackMessage) -> None:
        if message.is_initialized:
            # ロード済みなら処理せずに終了
            return
        message.is_initialized = True
        message._messages = []
        message._is_full = False
        try:
            if message.is_private:
                logger.debug(f"Do not get private conversations, channel_id={message.channel_id}, thread_ts={message.message.thread_ts}")
                return
            messages = self._integrate_duplicated_slack_file(
                self.slack_service.get_conversations_replies(message.channel_id, message.message.thread_ts, True)
            )
            if len(messages) == 0:
                # プライベートチャネルに対する読み取り権限がないなどの理由で空配列が返ってくるパターンがある
                # その場合は何もせずに終了する
                return
            if len(messages) <= 10:
                # スレッドの件数が 10件以内の場合はスレッド内の全メッセージを入力する
                # 起点メッセージを root_message に変更する
                message.message = messages[0]
                message._messages = messages[1:]
                message._root_message = None
                message._is_full = True
            elif len([v for v in messages if v.ts == message.message.ts]) >= 1:
                # 取得した最大60件のスレッド内に起点メッセージが含まれる場合（つまり先頭から60件以内に起点メッセージが存在する）
                for (i, v) in enumerate(messages):
                    if v.ts == message.message.ts:
                        break
                # reactions を取得するため起点メッセージを置き換える（search_message では reactions を取れない、conversations_replies は取れる）
                message.message = messages[i]
                if i == 0:
                    # 起点メッセージ == root_message だった場合、+9メッセージを入力
                    message._messages = messages[1:10]
                    message._root_message = None
                else:
                    # 起点メッセージ != root_message だった場合、起点メッセージは変更せず、root_message と、+8メッセージを入力
                    message._messages = messages[i + 1:i + 9]
                    message._root_message = messages[0]
            else:
                # 起点メッセージがスレッド内に含まれない場合、諦めて root_message だけ追加する
                message._root_message = messages[0]

            # 続けてファイルを読み込む
            for message_lite in [
                message.message,
                *([message._root_message] if message._root_message else []),
                *(message._messages if message._messages else []),
            ]:
                for file in message_lite.files:
                    # テキストファイルかcanvasだった場合はファイルをロードする
                    # 画像は読み込んでも利用しないので読み込まない
                    if file.is_text or file.is_canvas:
                        self.load_slack_file(file, True)
        except Exception as e:
            logger.error(e, exc_info=True)
