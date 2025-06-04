import dataclasses
import datetime
import json
import logging
import math
import re
from enum import Enum
from threading import RLock
from typing import Any, Optional

import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt.util.utils import get_boot_message
from slack_sdk.errors import SlackApiError
from slack_sdk.web.base_client import SlackResponse

from ossans_navi import config
from ossans_navi.common.cache import LRUCache
from ossans_navi.service.slack_wrapper import SlackWrapper
from ossans_navi.type.slack_type import (SlackAttachment, SlackChannel, SlackFile, SlackMessage, SlackMessageEvent, SlackMessageLite, SlackSearch,
                                         SlackSearchTerm, SlackUser)
from ossans_navi.type import ossans_navi_types

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

    @staticmethod
    def _thread_key(event: SlackMessageEvent) -> str:
        return f"{event.channel_id},{event.thread_ts}"

    def queue(self, event: SlackMessageEvent) -> None:
        with self:
            thread_key = EventGuard._thread_key(event)
            val = self.data.setdefault(thread_key, {})
            canceled_events = []
            for key in val.keys():
                if val[key].status in (EventGuard.Status.QUEUED, EventGuard.Status.RUNNING):
                    # 同一スレッド内で QUEUED, RUNNING のイベントはキャンセルする、スレッド内では同時に1つのイベントに対する処理しか実行しない
                    # キャンセルしたイベントを canceled_events に保持する、キャンセルしたイベントの内容も踏まえて処理するため
                    # 例: メッセージ1で @ossans_navi にメンションされ、メッセージ2 はメンションが無い場合でも @ossans_navi のメンションとして処理する
                    val[key].status = EventGuard.Status.CANCELED
                    canceled_events.extend(val[key].canceled_events)
                    if val[key].event.ts != event.ts:
                        # キューに入っているイベントの ts と今回のイベントの ts が一致する場合は更新イベントであり、元メッセージがイベントとして登録されている
                        # その場合は更新前の同一メッセージを canceled_events として扱わない、更新前のメッセージの情報は使う必要が無い
                        canceled_events.append(val[key].event)
            val[event.event_ts] = EventGuard.EventGuardData(EventGuard.Status.QUEUED, event, canceled_events)
            logger.info(f"EventGuard queued: {event.id()}")
            logger.info(f"EventGuard={self}")

    def start(self, event: SlackMessageEvent) -> None:
        with self:
            thread_key = EventGuard._thread_key(event)
            val = self.data.setdefault(thread_key, {})
            val[event.event_ts].status = EventGuard.Status.RUNNING
            logger.info(f"EventGuard started: {event.id()}")
            logger.info(f"EventGuard={self}")

    def cancel(self, event: SlackMessageEvent) -> None:
        """
        削除イベントの時だけ利用する
        削除された元イベントの情報をキューから探して CANCELED にする
        """
        with self:
            thread_key = EventGuard._thread_key(event)
            if thread_key not in self.data:
                return
            if event.ts not in self.data[thread_key]:
                return
            self.data[thread_key][event.ts].status = EventGuard.Status.CANCELED
            logger.info(f"EventGuard canceled: {event.id()}")
            logger.info(f"EventGuard={self}")

    def finish(self, event: SlackMessageEvent) -> None:
        with self:
            thread_key = EventGuard._thread_key(event)
            if thread_key not in self.data:
                return
            if event.event_ts not in self.data[thread_key]:
                return
            del self.data[thread_key][event.event_ts]
            if len(self.data[thread_key]) == 0:
                del self.data[thread_key]
            logger.info(f"EventGuard finished: {event.id()}")
            logger.info(f"EventGuard={self}")

    def is_queueed_or_running(self, event: SlackMessageEvent) -> bool:
        """
        イベントが QUEUEED or RUNNING かどうかを返す
        ただし更新イベントの場合は元イベントの QUEUEED or RUNNING かどうかを返す
        引数の event が更新イベントの場合は、そのイベントから元イベントを探して状態をチェックする
        EventGuard.is_canceled はそのイベント自体の状態を返却するので当メソッドと仕様が異なる
        """
        with self:
            thread_key = EventGuard._thread_key(event)
            if thread_key not in self.data:
                return False
            # 更新イベントでは event.ts が元イベントの ts を返却する、それによって更新イベントから元イベントの情報を参照できる
            if event.ts not in self.data[thread_key]:
                return False
            return self.data[thread_key][event.ts].status in (EventGuard.Status.QUEUED, EventGuard.Status.RUNNING)

    def is_canceled(self, event: SlackMessageEvent) -> bool:
        """
        イベントが CANCELED かどうかを返す
        更新イベントの場合でもそのイベント自体が CANCELED かどうかを返却する
        EventGuard.is_queueed_or_running は元イベントの状態を返却するので当メソッドと仕様が異なる
        """
        with self:
            thread_key = EventGuard._thread_key(event)
            if thread_key not in self.data:
                return False
            if event.event_ts not in self.data[thread_key]:
                return False
            return self.data[thread_key][event.event_ts].status == EventGuard.Status.CANCELED

    def terminate(self) -> None:
        with self:
            logger.info("EventGuard terminate")
            logger.info(f"EventGuard={self}")
            for (_, val) in self.data.items():
                for (_, data) in val.items():
                    data.status = EventGuard.Status.CANCELED
                    logger.info(f"EventGuard canceled: {data.event.id()}")

    def get_canceled_events(self, event: SlackMessageEvent) -> list[SlackMessageEvent]:
        with self:
            thread_key = EventGuard._thread_key(event)
            if thread_key not in self.data:
                return []
            if event.event_ts not in self.data[thread_key]:
                return []
            return self.data[thread_key][event.event_ts].canceled_events

    def __str__(self) -> str:
        return str(
            {
                thread_key: {
                    event_ts: {
                        "status": data.status.name,
                        "event_id": data.event.id(),
                        "canceled_events": [event.id() for event in data.canceled_events]
                    } for (event_ts, data) in val.items()
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
    def __init__(
        self,
        app_token: Optional[str] = None,
        user_token: Optional[str] = None,
        bot_token: Optional[str] = None
    ) -> None:
        self.app_token = app_token or config.SLACK_APP_TOKEN
        self.user_token = user_token or config.SLACK_USER_TOKEN
        self.bot_token = bot_token or config.SLACK_BOT_TOKEN
        self.app = App(token=self.bot_token, logger=logging.getLogger("slack_bolt"))
        self.socket_mode_hander = SocketModeHandler(self.app, self.app_token)
        self.app_client = SlackWrapper(token=self.app_token)
        self.user_client = SlackWrapper(token=self.user_token)
        self.bot_client = SlackWrapper(token=self.bot_token)
        self.cache_get_user = LRUCache[str, SlackUser](capacity=1000, expire=1 * 60 * 60 * 4)  # 4時間
        self.cache_get_bot = LRUCache[str, SlackUser](capacity=1000, expire=1 * 60 * 60 * 4)   # 4時間
        self.cache_get_conversations_members = LRUCache[str, list[str]](capacity=1000, expire=1 * 60 * 60 * 4)  # 4時間
        self.cache_get_channels = LRUCache[bool, dict[str, dict]](capacity=1, expire=1 * 60 * 60 * 4)   # 4時間
        self.cache_user_presence = LRUCache[str, bool](capacity=100, expire=1 * 60 * 10)   # 10分
        self.cache_get_channel = LRUCache[str, SlackChannel](capacity=1000, expire=1 * 60 * 60)   # 60分
        self.cache_config = LRUCache[bool, dict](capacity=1, expire=1 * 60 * 60)   # 60分
        self.cache_load_file = LRUCache[str, bytes](capacity=50, expire=1 * 60 * 60)   # 60分
        self.my_user_id: str = ""
        self.my_bot_user_id: str = ""
        self.my_bot_user: Optional[SlackUser] = None
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
        self.my_bot_user = self.get_user(response_bot["user_id"])
        self.workspace_url = response_bot["url"]
        logger.info(f"App start with: {response_app["app_name"]}")
        self.socket_mode_hander.connect()
        logger.info(get_boot_message())

    def stop(self) -> None:
        self.socket_mode_hander.close()

    def get_user(self, user_id: str) -> SlackUser:
        if (cached := self.cache_get_user.get(user_id)).found:
            return cached.value
        if re.search(r"^B", user_id):
            # B から始まる user_id はボットのものなのでボット用のAPIで取得する
            return self.get_bot(user_id)
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
            return cached.value
        try:
            response = self.bot_client.bots_info(bot=bot_id)
            response_bot: dict[str, Any] = response['bot']
            bot_user = SlackUser(
                user_id=response_bot.get("user_id", bot_id),
                name=response_bot.get('name', 'Unknown Bot'),
                username=response_bot.get('name', 'Unknown Bot'),
                mention=f"<@{bot_id}>",
                is_bot=True,
                is_guest=False,
                is_admin=False,
                bot_id=bot_id,
            )
        except SlackApiError as e:
            error_response: SlackResponse = e.response
            if error_response.get("error") != "bot_not_found":
                # error: bot_not_found ではない場合は例外を送出
                logger.error(f"{error_response.get("error")}, bot_id={bot_id}, exception={e}")
                logger.error(e, exc_info=True)
                raise e
            logger.info(f"{error_response.get("error")}, bot_id={bot_id}, exception={e}")
            bot_user = SlackUser(
                user_id=bot_id,
                name='Unknown Bot',
                username="unknown_bot",
                mention="",
                is_bot=True,
                is_guest=False,
                is_admin=False,
                is_valid=False,
                bot_id=bot_id,
            )
        except Exception as e:
            logger.error(f"bots_info returns error bot_id={bot_id}")
            logger.error(e, exc_info=True)
            raise e
        self.cache_get_bot.put(bot_id, bot_user)
        return bot_user

    def get_channel(self, channel_id: str, user_client: bool = False) -> SlackChannel:
        client: SlackWrapper = self.user_client if user_client else self.bot_client
        if (cached := self.cache_get_channel.get(channel_id)).found:
            return cached.value
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
            return cached.value
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
            return cached.value
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
            return cached.value
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
            files: list[dict[str, str | bool]] = message.get("files", [])
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
                files=[file for file in [SlackFile.from_dict(file) for file in files] if file is not None],
                reactions=([f":{reaction["name"]}:" for reaction in message["reactions"]] if "reactions" in message else [])
            ))
        return results

    def chat_post_message(
        self,
        channel: str,
        text: str,
        thread_ts: Optional[str] = None,
        images: list[ossans_navi_types.Image] = [],
    ) -> None:
        files: list[dict[str, Any]] = [
            {
                "filename": f"image{image.extension}",
                "content": image.data,
            }
            for image in images
        ]
        if files:
            self.bot_client.files_upload_v2(
                channel=channel,
                thread_ts=thread_ts,
                initial_comment=text,
                file_uploads=files,
            )
        else:
            self.bot_client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=text,
            )

    def add_reaction(self, channel: str, ts: str, names: list[str]) -> None:
        if not isinstance(names, list) or len(names) == 0:
            return
        name = re.sub(r'^:(.+):$', lambda v: v.group(1), names[0])
        try:
            self.bot_client.reactions_add(channel=channel, timestamp=ts, name=name)
        except SlackApiError as e:
            response: SlackResponse = e.response
            if response.get("error") == "invalid_name":
                logger.info("Reaction not found: " + name)
                if len(names) >= 2:
                    # OssansNavi が生成したリアクションが存在しない場合は次の候補を利用する
                    self.add_reaction(channel, ts, names[1:])
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
                    line = re.sub(r'^([ 　]*)[*-] ', lambda v: f"{v.group(1)!s}• ", line)
                    line = re.sub(r'^#{1,4} .+$', lambda v: f"*{v.group(0)!s}*", line)
                    line = re.sub(r' ?\*\*([^*]+)\*\* ?', lambda v: f" *{v.group(1)!s}* ", line)
                    line = re.sub(r' ?(?<!``)`([^`]+)`(?!``) ?', lambda v: f" `{v.group(1)!s}` ", line)
                    line = re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)', lambda x: f"<{x.group(2)!s}|{x.group(1)!s}>", line)
            else:
                if re.search(r'^```|```$', line):
                    is_codeblock = False
            lines.append(line)
        return "\n".join(lines)

    def _refine_slack_message(
        self,
        message: dict,
        recieved_message_user: SlackUser,
        recieved_message_channel_id: str,
        recieved_message_thread_ts: str,
        viewable_private_channels: list[str],
        trusted_bots: list[str],
    ) -> Optional[SlackMessage]:
        """
        slcka api の検索結果 dict を SlackMessage に変換する
        プライベートチャネルの権限確認も行う
        読み取り対象外のメッセージは None を返す
        """
        if (
            # 以下の条件のいずれかに合致する場合は該当メッセージは参照しない
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
        if (
            message_user.is_bot
            and message_user.user_id != self.my_bot_user_id
            and message_user.user_id not in trusted_bots
        ):
            # ボットの場合、OssansNavi 以外、かつ trusted_bots に入っていないなら情報源にしない
            # ボットのメッセージは情報源にしない理由
            #   ログメッセージなど大量のメッセージが引っかかり応答速度の低下してトークン数も浪費する可能性があるため
            # OssansNavi のメッセージは情報源とする理由
            #   ユーザーが OssansNavi へ画像を渡して情報を教えてくれるケースがある
            #   しかしユーザーのメッセージは画像のためキーワード検索ができない、一方で OssansNavi が返信したメッセージには画像内容も含めてテキスト化される
            #   そのため OssansNavi のメッセージを情報源とすれば、このパターンの情報も拾える
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
        files: list[dict[str, str | bool]] = message.get("files", [])
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
                files=[file for file in [SlackFile.from_dict(file) for file in files] if file is not None]
            ),
            channel=message["channel"]["name"],
            channel_id=message["channel"]["id"],
            score=message["score"] * score_multiply,
            is_private=message['channel']['is_private'],
        )

    def search(
        self,
        term: SlackSearchTerm,
        recieved_message_user: SlackUser,
        recieved_message_channel_id: str,
        recieved_message_thread_ts: str,
        viewable_private_channels: list[str],
        trusted_bots: list[str],
        is_additional: bool,
        is_get_messages: bool
    ) -> SlackSearch:
        results = self.user_client.search_messages(
            query=term.to_term(True),
            count=100,
            sort="score",
        )
        logger.debug(f"search_messages{"(additional)" if is_additional else ""}={{query={term.to_term(True)}, total={results["messages"]["total"]}}}")
        return SlackSearch(
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
                            viewable_private_channels,
                            trusted_bots,
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
    def duplicate_search_result(result: SlackSearch, year: int) -> SlackSearch:
        """指定した結果を指定の年数内のメッセージで絞り込んで新しい検索結果を生成する、同時に検索ワードに after:yyyy-mm-dd を付与する"""
        years_ago = datetime.datetime.now() - (datetime.timedelta(days=365) * year) - datetime.timedelta(days=1)
        term = SlackSearchTerm(result.term.words, years_ago, None)
        filtered_contents = [content for content in result.messages if content.message.timestamp > years_ago]
        return SlackSearch(
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
            return cached.value
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

    def load_file(self, url: str, user_client: bool = False) -> bytes:
        client: SlackWrapper = self.user_client if user_client else self.bot_client
        if not re.match(r'https?://(?:[^.]+\.)?slack.com/', url):
            raise ValueError(f"Not allowing host: {url}")
        if (cached := self.cache_load_file.get(url)).found:
            logger.info(f"Downloading(cache): {url}")
            return cached.value
        logger.info(f"Downloading: {url}")
        response = requests.get(url, headers={"Authorization": f"Bearer {client.token}"})
        if not (200 <= response.status_code < 300):
            raise RuntimeError(f"Response returns error: {response.status_code}")
        self.cache_load_file.put(url, response.content)
        return response.content

    def get_assistant_names(self) -> list[str]:
        return list({
            *(config.ASSISTANT_NAMES),
            *([self.my_bot_user.name] if self.my_bot_user else []),
        })
