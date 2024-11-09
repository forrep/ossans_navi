import dataclasses
import datetime
import html
import json
import logging
import math
import re
from collections import defaultdict
from enum import Enum
from io import BytesIO
from threading import RLock
from typing import Any, Callable, Optional

import requests
from PIL import Image, ImageFile
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt.util.utils import get_boot_message
from slack_sdk.errors import SlackApiError
from slack_sdk.web.base_client import SlackResponse

from ossans_navi import config
from ossans_navi.common.cache import LRUCache
from ossans_navi.service.slack_wrapper import SlackWrapper
from ossans_navi.type.slack_type import (SlackAttachment, SlackChannel, SlackFile, SlackMessage, SlackMessageEvent, SlackMessageLite, SlackSearch,
                                         SlackSearches, SlackSearchTerm, SlackUser)

MAX_IMAGE_SIZE = 1536

logger = logging.getLogger(__name__)


class EventGuard:
    class Status(Enum):
        QUEUED = 1
        RUNNING = 2
        CANCELED = 3

    @dataclasses.dataclass
    class EventGuardData:
        status: "EventGuard.Status"
        event: SlackMessageEvent
        canceled_events: list[SlackMessageEvent]

    def __init__(self) -> None:
        self.data: dict[str, dict[str, EventGuard.EventGuardData]] = {}
        self._lock = RLock()

    def queue(self, event: SlackMessageEvent) -> None:
        with self:
            thread_key = f"{event.channel_id()},{event.thread_ts()}"
            val = self.data.setdefault(thread_key, {})
            canceled_events = []
            for key in val.keys():
                val[key].status = EventGuard.Status.CANCELED
                canceled_events.append(val[key].event)
                canceled_events.extend(val[key].canceled_events)
            val[event.ts()] = EventGuard.EventGuardData(EventGuard.Status.QUEUED, event, canceled_events)
            logger.info(f"EventGuard queued: {event.id()}")
            logger.info(f"EventGuard={self}")

    def start(self, event: SlackMessageEvent) -> None:
        with self:
            thread_key = f"{event.channel_id()},{event.thread_ts()}"
            val = self.data.setdefault(thread_key, {})
            val[event.ts()].status = EventGuard.Status.RUNNING
            logger.info(f"EventGuard started: {event.id()}")
            logger.info(f"EventGuard={self}")

    def finish(self, event: SlackMessageEvent) -> None:
        with self:
            thread_key = f"{event.channel_id()},{event.thread_ts()}"
            if thread_key not in self.data:
                return
            if event.ts() not in self.data[thread_key]:
                return
            del self.data[thread_key][event.ts()]
            if len(self.data[thread_key]) == 0:
                del self.data[thread_key]
            logger.info(f"EventGuard finished: {event.id()}")
            logger.info(f"EventGuard={self}")

    def is_canceled(self, event: SlackMessageEvent) -> bool:
        with self:
            thread_key = f"{event.channel_id()},{event.thread_ts()}"
            if thread_key not in self.data:
                return False
            if event.ts() not in self.data[thread_key]:
                return False
            return self.data[thread_key][event.ts()].status == EventGuard.Status.CANCELED

    def get_canceled_events(self, event: SlackMessageEvent) -> list[SlackMessageEvent]:
        with self:
            thread_key = f"{event.channel_id()},{event.thread_ts()}"
            if thread_key not in self.data:
                return []
            if event.ts() not in self.data[thread_key]:
                return []
            return self.data[thread_key][event.ts()].canceled_events

    def __str__(self) -> str:
        return str(
            {
                thread_key: {
                    ts: {
                        "status": data.status.name,
                        "event_id": data.event.id(),
                        "canceled_events": [event.id() for event in data.canceled_events]
                    } for (ts, data) in val.items()
                }
                for (thread_key, val) in self.data.items()
            }
        )

    def __enter__(self):
        logger.debug("EventGuard lock acquire")
        return self._lock.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug(f"EventGuard lock release, {exc_type=}, {exc_value=}, {traceback=}")
        return self._lock.__exit__(exc_type, exc_value, traceback)


class SlackService:
    def __init__(self) -> None:
        self.app_token = config.SLACK_APP_TOKEN
        self.user_token = config.SLACK_USER_TOKEN
        self.bot_token = config.SLACK_BOT_TOKEN
        self.app = App(token=self.bot_token, logger=logging.getLogger("slack_bolt"))
        self.socket_mode_hander = SocketModeHandler(self.app, self.app_token)
        self.app_client = SlackWrapper(token=self.app_token)
        self.user_client = SlackWrapper(token=self.user_token)
        self.bot_client = SlackWrapper(token=self.bot_token)
        self.search_messages_exclude = ""
        self.cache_get_user = LRUCache[str, SlackUser](capacity=1000, expire=1 * 60 * 60 * 4)  # 4時間
        self.cache_get_bot = LRUCache[str, SlackUser](capacity=1000, expire=1 * 60 * 60 * 4)   # 4時間
        self.cache_get_conversations_members = LRUCache[str, list[str]](capacity=1000, expire=1 * 60 * 60 * 4)  # 4時間
        self.cache_get_channels = LRUCache[bool, dict[str, dict]](capacity=1, expire=1 * 60 * 60 * 4)   # 4時間
        self.cache_user_presence = LRUCache[str, bool](capacity=100, expire=1 * 60 * 10)   # 10分
        self.cache_get_channel = LRUCache[str, SlackChannel](capacity=1000, expire=1 * 60 * 60)   # 60分
        self.cache_config = LRUCache[bool, dict](capacity=1, expire=1 * 60 * 60)   # 60分
        self.my_user_id: str = ""
        self.my_bot_user_id: str = ""
        self.workspace_url: str = "https://slack.com/"

    def start(self) -> None:
        # 起動したトークンに紐づくアプリ名をロギングする、間違って本番トークンで起動してしまわないように
        response_app = self.app_client.auth_test()
        response_user = self.user_client.auth_test()
        response_bot = self.bot_client.auth_test()
        if not response_app.get("ok") or not response_user.get("ok") and not response_bot.get("ok"):
            logger.error(f"auth_test() returns error: app={str(response_app.data)}, user={str(response_user.data)}, bot={str(response_bot.data)}")
        self.my_user_id = response_user["user_id"]
        self.my_bot_user_id = response_bot["user_id"]
        self.workspace_url = response_bot["url"]
        self.search_messages_exclude = f"-from:<@{response_bot["user_id"]}>"
        logger.info(f"App start with: {response_app["app_name"]}")
        self.socket_mode_hander.connect()
        logger.info(get_boot_message())

    def stop(self) -> None:
        self.socket_mode_hander.close()

    def get_user(self, user_id: str) -> SlackUser:
        if (cached := self.cache_get_user.get(user_id)).found:
            return cached.value()
        try:
            response = self.bot_client.users_info(user=user_id)
            response_user: dict[str, Any] = response['user']
            # 別の Workspace から参加しているユーザーは is_stranger: true となっているので、is_guest とする（通常ユーザーは is_stranger 自体が送られない）
            slack_user = SlackUser(
                user_id=user_id,
                name=response_user.get("real_name", "Unknown"),
                username=response_user.get("name", "Unknown"),
                mention=f"<@{user_id}>",
                is_bot=bool(response_user['is_bot']),
                is_guest=bool(response_user.get("is_stranger", False) or response_user.get('is_restricted', True)),
                is_admin=bool(response_user.get('is_admin', False)),
            )
        except SlackApiError as e:
            error_response: SlackResponse = e.response
            if error_response.get("error") not in ("user_not_found", "user_not_visible"):
                # error: user_not_found, user_not_visible ではない場合は例外を送出
                logger.error(e, exc_info=True)
                raise e
            logger.info(f"{error_response.get("error")}, user_id={user_id}, exception={e}")
            slack_user = SlackUser(user_id, 'Unknown', 'Unknown', "", False, True, False, is_valid=False)
        except Exception as e:
            logger.error(f"users_info returns error, user_id={user_id}")
            logger.error(e, exc_info=True)
            raise e
        self.cache_get_user.put(user_id, slack_user)
        return slack_user

    def get_bot(self, bot_id: str) -> SlackUser:
        if (cached := self.cache_get_bot.get(bot_id)).found:
            return cached.value()
        try:
            response = self.bot_client.bots_info(bot=bot_id)
            response_bot: dict[str, Any] = response['bot']
            bot_user = SlackUser(
                user_id=bot_id,
                name=response_bot.get('name', 'Unknown Bot'),
                username=response_bot.get('name', 'Unknown Bot'),
                mention=f"<@{bot_id}>",
                is_bot=True,
                is_guest=False,
                is_admin=False,
            )
        except SlackApiError as e:
            error_response: SlackResponse = e.response
            if error_response.get("error") != "bot_not_found":
                # error: bot_not_found ではない場合は例外を送出
                # ・・・、する予定だけど、まずはエラーロギングだけして様子見
                logger.error(f"{error_response.get("error")}, bot_id={bot_id}, exception={e}")
                logger.error(e, exc_info=True)
                # raise e
            logger.info(f"{error_response.get("error")}, bot_id={bot_id}, exception={e}")
            bot_user = SlackUser(bot_id, 'Unknown Bot', "unknown_bot", "", True, False, False, is_valid=False)
        except Exception as e:
            logger.error(f"bots_info returns error bot_id={bot_id}")
            logger.error(e, exc_info=True)
            raise e
        self.cache_get_bot.put(bot_id, bot_user)
        return bot_user

    def get_channel(self, channel_id: str, user_client: bool = False) -> SlackChannel:
        client: SlackWrapper = self.user_client if user_client else self.bot_client
        if (cached := self.cache_get_channel.get(channel_id)).found:
            return cached.value()
        try:
            response = client.conversations_info(channel=channel_id)
            response_channel: dict[str, Any] = response["channel"]
            response_channel_topic: dict[str, Any] = response_channel.get("topic", {})
            response_channel_purpose: dict[str, Any] = response_channel.get("purpose", {})
            channel = SlackChannel(
                channel_id=channel_id,
                name=response_channel.get("name", ""),
                topic=response_channel_topic.get("value", ""),
                purpose=response_channel_purpose.get("value", ""),
                is_public=not response_channel.get("is_private") and not response_channel.get("is_im") and not response_channel.get("is_mpim"),
                is_private=bool(response_channel.get("is_private")),
                is_im=bool(response_channel.get("is_im")),
                is_mpim=bool(response_channel.get("is_mpim")),
            )
        except SlackApiError as e:
            error_response: SlackResponse = e.response
            if error_response.get("error") not in ("channel_not_found", ):
                # error: channel_not_found ではない場合は例外を送出
                # ・・・、する予定だけど、まずはエラーロギングだけして様子見
                logger.error(f"{error_response.get("error")}, channel_id={channel_id}, exception={e}")
                logger.error(e, exc_info=True)
                # raise e
            logger.info(f"{error_response.get("error")}, channel_id={channel_id}, exception={e}")
            channel = SlackChannel(
                channel_id=channel_id,
                name="Unknown",
                topic="",
                purpose="",
                is_public=False,
                is_private=False,
                is_im=False,
                is_mpim=False,
                is_valid=False,
            )
        except Exception as e:
            logger.error(f"conversations_info returns error, channel_id={channel_id}")
            logger.error(e, exc_info=True)
            raise e
        self.cache_get_channel.put(channel_id, channel)
        return channel

    def get_presence(self, user_id: str) -> bool:
        if (cached := self.cache_user_presence.get(user_id)).found:
            return cached.value()
        try:
            response = self.bot_client.users_getPresence(user=user_id)
            user_presence = response["presence"] == "active"
        except SlackApiError as e:
            error_response: SlackResponse = e.response
            if error_response.get("error") not in ("user_not_found", "user_not_visible"):
                # error: user_not_found, user_not_visible ではない場合は例外を送出
                logger.error(e, exc_info=True)
                raise e
            logger.info(f"{error_response.get("error")}, user_id={user_id}, exception={e}")
            user_presence = False
        except Exception as e:
            logger.error(f"users_getPresence returns error, user_id={user_id}")
            logger.error(e, exc_info=True)
            raise e
        self.cache_user_presence.put(user_id, user_presence)
        return user_presence

    def get_conversations_members(self, channel_id: str) -> list[str]:
        if (cached := self.cache_get_conversations_members.get(channel_id)).found:
            return cached.value()
        try:
            response = self.user_client.conversations_members(channel=channel_id, limit=1000)
            response_members: list[str] = response["members"]
        except SlackApiError as e:
            error_response: SlackResponse = e.response
            if error_response.get("error") != "channel_not_found":
                # error: channel_not_found ではない場合は例外を送出
                # ・・・、する予定だけど、まずはエラーロギングだけして様子見
                logger.error(f"{error_response.get("error")}, channel_id={channel_id}, exception={e}")
                logger.error(e, exc_info=True)
                # raise e
            logger.info(f"{error_response.get("error")}, channel_id={channel_id}, exception={e}")
            response_members = []
        except Exception as e:
            logger.error(f"conversations_members returns error channel_id={channel_id}")
            logger.error(e, exc_info=True)
            raise e
        self.cache_get_conversations_members.put(channel_id, response_members)
        return response_members

    def is_user_joined_channel(self, user: SlackUser, channel: str):
        return user.user_id in self.get_conversations_members(channel)

    def get_channels(self) -> dict[str, dict]:
        if (cached := self.cache_get_channels.get(True)).found:
            return cached.value()
        try:
            response = self.bot_client.conversations_list(limit=1000)
            response_channels = response["channels"]
            channels = {
                channel["name"]: {
                    "name": channel["name"],
                    "num_members": channel["num_members"],
                    "score": (math.log2(channel["num_members"]) if channel["num_members"] >= 1 else 0.0) + 1.0
                }
                for channel in response_channels
            }
        except Exception as e:
            logger.error("conversations_list returns error")
            logger.error(e, exc_info=True)
            raise e
        self.cache_get_channels.put(True, channels)
        return channels

    def replace_id_to_name(self, text) -> str:
        if not isinstance(text, str):
            return text
        text = re.sub(
            r'<#([A-Z0-9]+)(?:\|[^>]*?)?>(\s)?',
            lambda v: f"#{self.get_channel(v.group(1), True).name}{v.group(2) if v.group(2) else " "}",
            text
        )
        text = re.sub(
            r'<@([A-Z0-9]+)(?:\|[^>]+?)?>( *さん)?',
            lambda v: self.get_user(v.group(1)).name + "さん",
            text
        )
        return text

    def disable_mention_if_not_active(self, text) -> str:
        if not isinstance(text, str):
            return text
        return re.sub(
            r'<@([A-Z0-9]+)>',
            lambda v: v.group(0) if self.get_presence(v.group(1)) else ("@" + self.get_user(v.group(1)).username),
            text
        )

    def _get_content_callable(self, token) -> Callable[[SlackFile], None]:
        def _get_content(file: SlackFile) -> None:
            file.get_content = None
            file._content = None
            try:
                file._content = SlackService.get_file(token, file.link)
                if file.is_image():
                    bytes_io = BytesIO(file._content)
                    image: Image.Image | ImageFile.ImageFile = Image.open(bytes_io)
                    max_size = max(image.width, image.height)
                    if max_size > MAX_IMAGE_SIZE:
                        resize_retio = MAX_IMAGE_SIZE / max_size
                        image = image.resize((int(image.width * resize_retio), int(image.height * resize_retio)))
                        bytes_io = BytesIO()
                        image.save(bytes_io, format="PNG")
                        file._content = bytes_io.getvalue()
                        file.mimetype = 'image/png'
                    file._height = image.height
                    file._width = image.width
                if file.is_text():
                    file.text = file._content.decode("utf-8")
                    if file.mimetype == "text/html":
                        # 数値文字参照を通常の文字列に変換
                        file.text = html.unescape(file.text)
            except Exception as e:
                logger.info(f"Slack get_image failed: {str(e)}")
        return _get_content

    def get_conversations_replies(self, channel: str, thread_ts: str, user_client: bool = False) -> list[SlackMessageLite]:
        client: SlackWrapper = self.user_client if user_client else self.bot_client
        try:
            response = client.conversations_replies(channel=channel, ts=thread_ts, limit=60)
            response_messages: list[dict] = response["messages"]
        except SlackApiError as e:
            error_response: SlackResponse = e.response
            if user_client and error_response.get("error") == "missing_scope":
                # user_token で実行する search_message はアプリインストール者の権限に準ずるためプライベートチャネルやDMもヒットする
                # 一方で付与された scope はパブリックチャネルしか読み取れないので、missing_scope が発生する
                # よって、user_client で発生する missing_scope は想定内
                logger.info(f"{error_response.get("error")}, channel={channel}, thread_ts={thread_ts}, user_client={user_client}, exception={e}")
                return []
            else:
                logger.error(f"{error_response.get("error")}, channel={channel}, thread_ts={thread_ts}, user_client={user_client}, exception={e}")
                raise e
        except Exception as e:
            logger.error(f"conversations_replies returns error, channel={channel}, thread_ts={thread_ts}, user_client={user_client}, exception={e}")
            raise e
        results = []
        for message in response_messages:
            attachments_dict: list[dict[str, str]] = message.get("attachments", [])
            attachments: list[SlackAttachment] = []
            for attachment_dict in attachments_dict:
                attachment = SlackAttachment.from_dict(attachment_dict)
                attachment.user = self.get_user(attachment_dict["author_id"]) if attachment_dict.get("author_id", "").startswith("U") else None
                attachments.append(attachment)
            files: list[dict[str, str]] = message.get("files", [])
            results.append(SlackMessageLite(
                timestamp=datetime.datetime.fromtimestamp(float(message["ts"])),
                ts=message["ts"],
                thread_ts=thread_ts,
                user=self.get_bot(message['bot_id']) if not message.get('user') and message.get("bot_id") else self.get_user(message['user']),
                content=self.replace_id_to_name(message["text"]),
                # thread_ts と ts が異なる場合はスレッド内の子メッセージ
                permalink=(
                    f"{self.workspace_url}archives/{channel}/p{thread_ts.replace('.', '')}"
                    + (f"?thread_ts={message["thread_ts"]}" if thread_ts != message["ts"] else "")
                ),
                attachments=attachments,
                files=[SlackFile.from_dict(file, self._get_content_callable(self.bot_token)) for file in files],
                reactions=([f":{reaction["name"]}:" for reaction in message["reactions"]] if "reactions" in message else [])
            ))
        return results

    def chat_post_message(self, channel: str, text: str, thread_ts: Optional[str] = None) -> None:
        self.bot_client.chat_postMessage(channel=channel, text=text, thread_ts=thread_ts)

    def add_reaction(self, channel: str, ts: str, name: str | None, fallback: str | None = None) -> None:
        if not isinstance(name, str):
            return
        name = re.sub(r'^:(.+):$', lambda v: v.group(1), name)
        try:
            self.bot_client.reactions_add(channel=channel, timestamp=ts, name=name)
        except SlackApiError as e:
            response: SlackResponse = e.response
            if response.get("error") == "invalid_name":
                logger.info("Reaction not found: " + name)
                if fallback:
                    # OssansNavi が生成したリアクションが存在しない場合は fallback を利用する
                    self.add_reaction(channel, ts, fallback)
                    return
            logger.error("reactions_add() return error.")
            logger.error(e, exc_info=True)

    @staticmethod
    def convert_markdown_to_mrkdwn(text: str):
        is_codeblock = False
        lines = []
        for line in text.split("\n"):
            if not is_codeblock:
                if (matcher := re.match(r'^```(?:[a-z_]{,13})?(.*)$', line)):
                    is_codeblock = True
                    line = "```" + ("\n" + matcher.group(1) if matcher.group(1) else "")
                else:
                    line = re.sub(r'^([ 　]*)- ', lambda v: f"{v.group(1)}• ", line)
                    line = re.sub(r'^#{1,4} .+$', lambda v: f"*{v.group(0)}*", line)
                    line = re.sub(r' ?\*\*([^*]+)\*\* ?', lambda v: f" *{v.group(1)}* ", line)
                    line = re.sub(r' ?(?<!``)`([^`]+)`(?!``) ?', lambda v: f" `{v.group(1)}` ", line)
                    line = re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)', lambda x: f"<{x.group(2)}|{x.group(1)}>", line)
            else:
                if re.search(r'^```|```$', line):
                    is_codeblock = False
            lines.append(line)
        return "\n".join(lines)

    def _get_messages_callable(self) -> Callable[[SlackMessage], None]:
        def _get_messages(message: SlackMessage) -> None:
            message.get_messages = None
            message._messages = []
            message._is_full = False
            try:
                if message.is_private:
                    logger.debug(f"Do not get private conversations, channel_id={message.channel_id}, thread_ts={message.message.thread_ts}")
                    return
                messages = self.get_conversations_replies(message.channel_id, message.message.thread_ts, True)
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
            except SlackApiError as e:
                response: SlackResponse = e.response
                if response.get("error") == "missing_scope":
                    logger.info(f"Do not get private channel message: channel_id={message.channel_id}, thread_ts={message.message.thread_ts}")
                else:
                    logger.error(e, exc_info=True)
            except Exception as e:
                logger.error(e, exc_info=True)
        return _get_messages

    def _refine_slack_message(
        self,
        message: dict,
        recieved_message_user: SlackUser,
        recieved_message_channel_id: str,
        recieved_message_thread_ts: str,
        viewable_private_channels: list[str],
    ) -> SlackMessage | None:
        """
        slcka api の検索結果 dict を SlackMessage に変換する
        プライベートチャネルの権限確認も行う
        読み取り対象外のメッセージは None を返す
        """
        if (
            # 以下の条件に合致する場合は該当メッセージは参照しない
            # セーフモードがONで、かつプライベートチャネルで、かつ safe_mode_viewable_channels に含まれていないチャネルの場合
            (config.SAFE_MODE and message['channel']['is_private'] and message['channel']['id'] not in viewable_private_channels)
            # プライベートチャネル、かつ元メッセージ送信者がそのチャネルに参加していない場合
            or message['channel']['is_private'] and not self.is_user_joined_channel(recieved_message_user, message['channel']['id'])
            # DM ※情報価値が低いことが多い
            or message['channel']['is_im']
            # グループDM ※情報価値が低いことが多い
            or message['channel']['is_mpim']
            # 投稿者が null ※RSSフィードなどが該当する、ワークスペース固有の情報ではないため入力しない、LMM 自体の学習結果に期待する
            or message["user"] is None
            # 開発用チャネル ※テスト用にノイズとなる質問が多数投稿されているため
            or message["channel"]["id"] in config.DEVELOPMENT_CHANNELS
        ):
            return None
        message_user = self.get_user(message["user"])
        if message_user.is_bot:
            # ボットのメッセージは情報源にしない
            return None
        if not (
            matcher := re.match(
                r'https://([^.]+\.)?slack\.com/archives/(?:[^/]+)/p([0-9]+)([0-9]{6})(?:\?thread_ts=([0-9.]+))?',
                message["permalink"]
            )
        ):
            logger.error("Permalink not match: " + message["permalink"])
            return None
        ts = f"{matcher.group(2)}.{matcher.group(3)}"
        ts_without_dot = f"{matcher.group(2)}{matcher.group(3)}"
        subdomain = matcher.group(1) if matcher.group(1) else ""
        thread_ts = matcher.group(4)
        channel_id = message["channel"]["id"]
        # thread_ts が指定されていないか、 ts と thread_ts が同一ならスレッドの親メッセージ
        is_root_message = thread_ts is None or ts == thread_ts
        # デフォルトの permalink はスレッドの root_message でも thread_ts が付いてトークンの無駄なので再構築する、permalink を一意にするのとトークン節約のため
        permalink = f'https://{subdomain}slack.com/archives/{channel_id}/p{ts_without_dot}{"?thread_ts=" + thread_ts if not is_root_message else ""}'
        if recieved_message_channel_id == channel_id and recieved_message_thread_ts in (ts, thread_ts):
            # やりとりしているメッセージと同一スレッドは当然検索結果に引っかかるが、役に立たないのでスルーする
            return None
        score_multiply = 1.0
        if (channel := self.get_channels().get(message["channel"]["name"])):
            score_multiply = channel["score"]
        elif (num_members := len(self.get_conversations_members(message["channel"]["id"]))) >= 1:
            score_multiply = math.log2(num_members) + 1.0
        attachments: list[dict[str, str]] = message.get("attachments", [])
        files: list[dict[str, str]] = message.get("files", [])
        return SlackMessage(
            message=SlackMessageLite(
                timestamp=datetime.datetime.fromtimestamp(float(message["ts"])),
                ts=ts,
                thread_ts=thread_ts if thread_ts else ts,
                user=message_user,
                content=message["text"],
                permalink=permalink,
                attachments=[
                    SlackAttachment.from_dict(attachment)
                    for attachment in attachments
                ],
                files=[SlackFile.from_dict(file, self._get_content_callable(self.bot_token)) for file in files]
            ),
            channel=message["channel"]["name"],
            channel_id=message["channel"]["id"],
            score=message["score"] * score_multiply,
            is_private=message['channel']['is_private'],
            get_messages=self._get_messages_callable(),
        )

    def _search(
        self,
        term: SlackSearchTerm,
        recieved_message_user: SlackUser,
        recieved_message_channel_id: str,
        recieved_message_thread_ts: str,
        viewable_private_channels: list[str],
        is_additional: bool,
        is_get_messages: bool
    ) -> SlackSearch:
        results = self.user_client.search_messages(
            query=f"{term.to_term(True)}{' ' + self.search_messages_exclude if self.search_messages_exclude else ''}",
            count=100,
            sort="score",
        )
        logger.debug(f"search_messages{"(additional)" if is_additional else ""}={{query={term.to_term(True)}, total={results["messages"]["total"]}}}")
        return SlackSearch(
            words=term.to_term(False),
            term=term,
            total_count=results["messages"]["total"],
            messages=sorted(
                [
                    message for message in [
                        self._refine_slack_message(
                            message,
                            recieved_message_user,
                            recieved_message_channel_id,
                            recieved_message_thread_ts,
                            viewable_private_channels
                        )
                        for message in results['messages']['matches']
                    ] if message is not None
                ],
                key=lambda v: v.score, reverse=True
            ),
            is_full=results["messages"]["total"] == results["messages"]["pagination"]["last"],
            is_additional=is_additional,
            is_get_messages=is_get_messages,
        )

    @staticmethod
    def _validate_words(words: list[str]) -> list[str]:
        results = []
        for word in words:
            if re.search(r'\bfrom:(?!<@[A-Z0-9]+>)', word):
                continue
            results.append(word)
        return results

    def search(
        self,
        slack_searches: SlackSearches,
        terms: list[str],
        recieved_message_user: SlackUser,
        recieved_message_channel_id: str,
        recieved_message_thread_ts: str,
        viewable_private_channels: list[str],
        is_additional: bool = False,
        is_get_messages: bool = False
    ) -> None:
        terms = SlackService._validate_words(terms)
        terms_dict: defaultdict[frozenset[str], set[SlackSearchTerm]] = defaultdict(set)
        for term in [SlackSearchTerm.parse(term) for term in terms]:
            terms_dict[frozenset(term.words)].add(term)

        # 絞り込みのキーワード数が少なく、文字数も少ない検索条件から順番に検索する
        # なぜならば「ワードA AND ワードB」の検索結果が 10件ならば 「ワードA AND ワードB AND ワードC」の検索は実行の必要がないからスキップできるように
        for current_term_words in sorted(terms_dict.keys(), key=lambda v: (len(v), sum([len(w) for w in v]),)):
            current_terms = sorted(
                terms_dict[current_term_words],
                key=lambda v: (v.date_from.timestamp() if v.date_from else 0.0, (1 / v.date_to.timestamp()) if v.date_to else 0.0)
            )
            for current_term in current_terms:
                if (
                    any(
                        [
                            (result.is_full and result.term.is_subset(current_term)) or result.term == current_term
                            for result in slack_searches.results
                        ]
                    )
                ):
                    # 検索済み結果に今回の検索条件より緩く、かつ全検索結果を取得済みならば current_term は検索の必要が無いのでスキップする
                    continue
                result = self._search(
                    current_term,
                    recieved_message_user,
                    recieved_message_channel_id,
                    recieved_message_thread_ts,
                    viewable_private_channels,
                    is_additional,
                    is_get_messages,
                )
                slack_searches.add(result)
                if result.term.date_from is None and result.term.date_to is None:
                    if result.is_meny_messages():
                        slack_searches.add(SlackService._duplicate_search_result(result, 2))
                    if result.is_too_meny_messages():
                        slack_searches.add(SlackService._duplicate_search_result(result, 1))

    @staticmethod
    def _duplicate_search_result(result: SlackSearch, year: int) -> SlackSearch:
        """指定した結果を指定の年数内のメッセージで絞り込んで新しい検索結果を生成する、同時に検索ワードに after:yyyy-mm-dd を付与する"""
        years_ago = datetime.datetime.now() - (datetime.timedelta(days=365) * year) - datetime.timedelta(days=1)
        term = SlackSearchTerm(result.term.words, years_ago, None)
        filtered_contents = [content for content in result.messages if content.message.timestamp > years_ago]
        return SlackSearch(
            words=term.to_term(),
            term=term,
            total_count=result.total_count * len(filtered_contents) // len(result.messages),
            messages=filtered_contents,
            is_full=result.is_full,
            is_additional=result.is_additional,
            is_get_messages=result.is_get_messages,
        )

    def get_dm_info_with_ossans_navi(self, user_id: str) -> str:
        channel: dict = self.bot_client.conversations_open(users=user_id).get("channel", {})
        channel_id = channel.get("id")
        if not channel_id:
            return ""
        channel = self.bot_client.conversations_info(channel=channel_id).get("channel", {})
        topic_dict: dict = channel.get("topic", {})
        return topic_dict.get("value", "")

    def store_config_dict(self, config_dict: dict) -> None:
        channel: dict = self.bot_client.conversations_open(users="USLACKBOT").get("channel", {})
        channel_id = channel.get("id")
        if not channel_id:
            raise RuntimeError('failed to store config, slackbot channel not found.')
        self.bot_client.chat_postMessage(channel=channel_id, text=json.dumps({"type": "config", **config_dict}, ensure_ascii=False))
        self.cache_config.clear(True)

    def get_config_dict(self) -> dict:
        if (cached := self.cache_config.get(True)).found:
            return cached.value()
        config_dict_default: dict = {"type": "config"}
        channel: dict = self.bot_client.conversations_open(users="USLACKBOT").get("channel", {})
        channel_id = channel.get("id")
        if not channel_id:
            self.cache_config.put(True, config_dict_default)
            return config_dict_default
        messages: list[dict[str, str]] = self.bot_client.conversations_history(channel=channel_id, limit=100).get("messages", [])
        for (i, message) in enumerate(messages):
            message_text = message.get("text")
            if message_text is None:
                continue
            try:
                config_dict: dict = json.loads(message_text)
                if config_dict.get("type") == "config":
                    self.cache_config.put(True, config_dict)
                    if i >= 10:
                        self.store_config_dict(config_dict)
                    return config_dict
            except Exception:
                # JSONでパースできない場合は例外になるがエラーではない
                # 当アプリが Slackbot とのメッセージに設定JSONを保存しているだけなので、設定JSON以外のテキストも普通に返ってくる
                pass
        self.cache_config.put(True, config_dict_default)
        return config_dict_default

    @staticmethod
    def get_file(token: str, url: str) -> bytes:
        if not re.match(r'https?://(?:[^.]+\.)?slack.com/', url):
            raise ValueError(f"Not allowing host: {url}")
        logger.info(f"Downloading: {url}")
        response = requests.get(url, headers={"Authorization": f"Bearer {token}"})
        if not (200 <= response.status_code < 300):
            raise RuntimeError(f"Response returns error: {response.status_code}")
        return response.content
