import base64
import dataclasses
import datetime
import hashlib
import itertools
import re
from threading import RLock
from typing import Any, Callable, Iterable, Optional


@dataclasses.dataclass
class SlackUser:
    user_id: str
    name: str
    username: str
    mention: str
    is_bot: bool
    is_guest: bool
    is_admin: bool
    is_valid: bool = dataclasses.field(default=True)


@dataclasses.dataclass
class SlackAttachment:
    title: str
    text: str
    link: str
    user: Optional[SlackUser] = dataclasses.field(default=None, init=False)

    @staticmethod
    def from_dict(attachment: dict[str, str]) -> 'SlackAttachment':
        return SlackAttachment(
            title=attachment.get("title", ""),
            text=attachment.get("text", ""),
            link=attachment.get("title_link", attachment.get("from_url", "")),
        )


@dataclasses.dataclass
class SlackChannel:
    channel_id: str
    name: str
    topic: str
    purpose: str
    is_public: bool
    is_private: bool
    is_im: bool
    is_mpim: bool
    is_valid: bool = dataclasses.field(default=True)


@dataclasses.dataclass
class SlackFile:
    title: str
    mimetype: str
    link: str
    get_content: Optional[Callable[['SlackFile'], None]]
    is_analyzed: bool = dataclasses.field(default=False, init=False)
    description: Optional[str] = dataclasses.field(default=None, init=False)
    text: Optional[str] = dataclasses.field(default=None, init=False)
    _content: Optional[bytes] = dataclasses.field(default=None, init=False)
    _height: int = dataclasses.field(default=0, init=False)
    _width: int = dataclasses.field(default=0, init=False)

    @staticmethod
    def from_dict(file: dict[str, str], get_content: Callable[['SlackFile'], None]) -> 'SlackFile':
        slack_file = SlackFile(
            title=file.get("title", ""),
            mimetype=file.get("mimetype", ""),
            link=file.get("url_private", ""),
            get_content=get_content
        )
        if slack_file.is_text() and file.get("plain_text"):
            # text で plain_text があればそれを利用する
            slack_file.get_content = None
            slack_file.text = file["plain_text"]
        return slack_file

    def initialize(self) -> None:
        if callable(self.get_content):
            self.get_content(self)

    def content(self) -> bytes:
        if callable(self.get_content):
            self.get_content(self)
        if self._content is None:
            raise ValueError('self._content is None')
        return self._content

    def is_image(self) -> bool:
        return isinstance(self.mimetype, str) and self.mimetype.startswith("image/")

    def is_text(self) -> bool:
        return isinstance(self.mimetype, str) and self.mimetype.startswith("text/")

    def is_textualize(self) -> bool:
        return self.is_text() or (self.is_image() and self.is_analyzed)

    def width(self) -> int:
        if callable(self.get_content):
            self.get_content(self)
        return self._width

    def height(self) -> int:
        if callable(self.get_content):
            self.get_content(self)
        return self._height

    def valid(self) -> bool:
        try:
            return bool(self.content())
        except Exception:
            return False

    def image_uri(self) -> str:
        return f"data:image/png;base64,{base64.b64encode(self.content()).decode()}"

    def to_dict(self) -> dict:
        if callable(self.get_content):
            self.get_content(self)
        return {
            "title": self.title,
            "mimetype": self.mimetype,
            "link": self.link,
            **(
                {"width": self.width(), "height": self.height()} if self.is_image() else {}
            ),
            **(
                {"description": self.description} if self.description else {}
            ),
            **(
                {"text": self.text} if self.text else {}
            ),
        }


@dataclasses.dataclass
class SlackMessageLite:
    timestamp: datetime.datetime
    ts: str
    thread_ts: str
    user: SlackUser
    content: str
    permalink: str
    attachments: list[SlackAttachment]
    files: list[SlackFile]
    reactions: list[str] = dataclasses.field(default_factory=list)

    def is_root_message(self) -> bool:
        return self.ts == self.thread_ts

    def has_files(self) -> bool:
        return len(self.files) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "ts": self.ts,
            "thread_ts": self.thread_ts,
            "user": dataclasses.asdict(self.user),
            "content": self.content,
            "permalink": self.permalink,
            "attachments": [dataclasses.asdict(attachment) for attachment in self.attachments],
            "files": [file.to_dict() for file in self.files],
        }


@dataclasses.dataclass
class SlackMessage:
    message: SlackMessageLite
    channel: str
    channel_id: str
    score: float
    is_private: bool
    get_messages: Optional[Callable[['SlackMessage'], None]]
    _root_message: Optional[SlackMessageLite] = dataclasses.field(default=None, init=False)
    _messages: Optional[list[SlackMessageLite]] = dataclasses.field(default=None, init=False)
    _is_full: bool = dataclasses.field(default=False, init=False)

    def is_initialized(self) -> bool:
        return not callable(self.get_messages)

    def initialize(self) -> None:
        if callable(self.get_messages):
            self.get_messages(self)
        for message in [
            self.message,
            *([self._root_message] if self._root_message else []),
            *(self._messages if self._messages else []),
        ]:
            for file in message.files:
                # テキストファイルだった場合はファイルのロード
                # 画像の場合は読み込んでも利用できないので読み込まない
                if file.is_text():
                    file.initialize()

    def root_message(self) -> SlackMessageLite | None:
        self.initialize()
        return self._root_message

    def messages(self) -> list[SlackMessageLite]:
        self.initialize()
        if self._messages is None:
            raise ValueError('self._messages is None')
        return self._messages

    def is_full(self) -> bool:
        self.initialize()
        return self._is_full

    @staticmethod
    def sort(messages: list['SlackMessage']) -> list['SlackMessage']:
        return sorted(messages, key=lambda v: v.message.timestamp, reverse=True)


@dataclasses.dataclass
class SlackSearch:
    words: str
    total_count: int
    messages: list[SlackMessage]
    is_additional: bool
    is_get_messages: bool

    def get_id(self) -> str:
        return hashlib.md5(self.words.encode('utf8')).hexdigest()


@dataclasses.dataclass
class SlackSearches:
    results: list[SlackSearch] = dataclasses.field(default_factory=list, init=False)
    messages: dict[str, SlackMessage] = dataclasses.field(default_factory=dict, init=False)
    total_count: int = dataclasses.field(default=0, init=False)
    used: set[str] = dataclasses.field(default_factory=set, init=False)
    lastshot: dict[str, SlackMessage] = dataclasses.field(default_factory=dict, init=False)
    _lock: RLock = dataclasses.field(default_factory=RLock, init=False)
    _lastshot_permalinks: set[str] = dataclasses.field(default_factory=set, init=False)

    def add(self, results: list[SlackSearch]) -> None:
        for result in results:
            # 同一 permalink の SlackMessage は 1つのインスタンスにまとめる
            result.messages = [self.messages.setdefault(message.message.permalink, message) for message in result.messages]
        self.results = sorted([*self.results, *results], key=lambda v: v.total_count)
        self.total_count = sum([len(v.messages) for v in self.results])

    def __iter__(self):
        return iter(self.results)

    def result_len(self) -> int:
        return len(self.results)

    def use(self, permalinks: list[str] | str) -> None:
        if isinstance(permalinks, list):
            self.used.update(permalinks)
        if isinstance(permalinks, str):
            self.used.add(permalinks)

    def is_used(self, permalink: str) -> bool:
        return permalink in self.used

    def add_lastshot(self, permalinks: list[str]) -> None:
        """
        lastshot で RAG の結果として入力するメッセージのリストに追加する
        ただし、すでにメッセージが追加済みの場合は二重で追加しない
        """
        for permalink in permalinks:
            self._add_lastshot(permalink)

    def _add_lastshot(self, permalink: str) -> None:
        if permalink in self._lastshot_permalinks:
            # すでに lastshot に追加済みなので終了
            return
        for result in self.results:
            # 検索結果の中から起点メッセージだけを探索する
            for message in result.messages:
                if not message.is_initialized():
                    # initialize() されていないメッセージはまだ処理対象外なので確認の必要はない
                    # そのメッセージが initialize() されたタイミングで確認される
                    continue
                if permalink == message.message.permalink:
                    # lastshot に追加したい message が見つかった
                    self.lastshot[message.message.permalink] = message
                    self._lastshot_permalinks.add(message.message.permalink)
                    if message.is_full():
                        # そのメッセージにスレッド内の全メッセージが含まれている場合はそのスレッド内の全メッセージの permalink を 追加済みとしてマークする
                        # なぜならそのメッセージにはスレッド内の全メッセージが含まれているので、追加で別メッセージを読み込む必要がないため
                        self._lastshot_permalinks.update([v.permalink for v in message.messages()])
                        if (v := message.root_message()):
                            self._lastshot_permalinks.add(v.permalink)
                    return
        for result in self.results:
            # root_message とスレッド内のメッセージを探索する
            for message in result.messages:
                if not message.is_initialized():
                    # initialize() されていないメッセージはまだ処理対象外なので確認の必要はない
                    # そのメッセージが initialize() されたタイミングで確認される
                    continue
                if (
                    ((v := message.root_message()) and permalink == v.permalink)
                    or (permalink in [v.permalink for v in message.messages()])
                ):
                    # root_message またはスレッド内のメッセージの permalink と一致
                    self.lastshot[message.message.permalink] = message
                    self._lastshot_permalinks.add(message.message.permalink)
                    if message.is_full():
                        # そのメッセージにスレッド内の全メッセージが含まれている場合はそのスレッド内の全メッセージの permalink を 追加済みとしてマークする
                        # なぜならそのメッセージにはスレッド内の全メッセージが含まれているので、追加で別メッセージを読み込む必要がないため
                        self._lastshot_permalinks.update([v.permalink for v in message.messages()])
                        if (v := message.root_message()):
                            self._lastshot_permalinks.add(v.permalink)
                    return

    def get_lastshot(self) -> Iterable[SlackMessage]:
        return self.lastshot.values()

    def is_searched(self, words: str) -> bool:
        return words in [result.words for result in self.results]

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self._lock.__exit__(exc_type, exc_value, traceback)


@dataclasses.dataclass
class SlackMessageEvent:
    source: dict[str, Any]
    _user: Optional[SlackUser] = dataclasses.field(default=None, init=False)
    _channel: Optional[SlackChannel] = dataclasses.field(default=None, init=False)
    is_mention: bool = dataclasses.field(default=False, init=False)
    is_talk_to_other: bool = dataclasses.field(default=False, init=False)
    is_joined: bool = dataclasses.field(default=False, init=False)
    is_next_message_from_ossans_navi: bool = dataclasses.field(default=False, init=False)
    classification: str = dataclasses.field(default="other", init=False)
    settings: str = dataclasses.field(default="", init=False)
    canceled_events: list['SlackMessageEvent'] = dataclasses.field(default_factory=list, init=False)

    def valid(self) -> bool:
        return bool(self._user)

    @property
    def user(self) -> SlackUser:
        if self._user:
            return self._user
        raise ValueError("self._user is None")

    @user.setter
    def user(self, value: SlackUser):
        self._user = value

    @property
    def channel(self) -> SlackChannel:
        if self._channel:
            return self._channel
        raise ValueError("self._channel is None")

    @channel.setter
    def channel(self, value: SlackChannel):
        self._channel = value

    def channel_id(self) -> str:
        return self.source["channel"]

    def text(self) -> str:
        return self.source["text"]

    def user_id(self) -> str:
        return self.source["user"]

    def ts(self) -> str:
        return self.source["ts"]

    def thread_ts(self) -> str:
        # スレッドの場合はスレッドの大元メッセージの ts を保持する thread_ts を利用、スレッドではない場合はそのメッセージからスレッドを作るので ts を利用
        return self.source["thread_ts"] if "thread_ts" in self.source else self.source["ts"]

    def timestamp(self) -> str:
        return datetime.datetime.fromtimestamp(float(self.ts())).strftime('%Y-%m-%d %H:%M:%S')

    def _mentions(self) -> list[str]:
        return [user_id[0] for user_id in re.findall(r'<@([A-Z0-9]+)(\|[^>]+?)?>', self.text())]

    def mentions(self) -> list[str]:
        return [
            *(self._mentions()),
            *(
                itertools.chain.from_iterable(
                    [event._mentions() for event in self.canceled_events]
                )
            )
        ]

    def _is_broadcast(self) -> bool:
        return bool(re.search(r'<!(?:channel|here)>', self.text()))

    def is_broadcast(self) -> bool:
        return any([
            self._is_broadcast(),
            *[event._is_broadcast() for event in self.canceled_events]
        ])

    def is_reply_to_ossans_navi(self) -> bool:
        return not self.is_talk_to_other and self.is_next_message_from_ossans_navi

    def is_thread(self) -> bool:
        return 'thread_ts' in self.source

    def is_message_post(self) -> bool:
        """
        応答が必要なメッセージ（1, 2 の条件に当てはまる）に True を返却
        1. text, channel, user, ts が存在する
        2. 次のいずれかに当てはまる
            - subtype が存在しない → 通常メッセージ
            - subtype が file_share → テキストスニペットの送信
            - subtype が thread_broadcast のいずれか → チャネルにも投稿するチェックを入れてスレッド返信
        """
        return (
            all(v in self.source for v in ("text", "channel", "user", "ts", ))
            and (
                "subtype" not in self.source
                or self.source["subtype"] in ("file_share", "thread_broadcast")
            )
        )

    def is_open_channel(self) -> bool:
        return self.source.get("channel_type") == "channel"

    def is_dm(self) -> bool:
        return self.source.get("channel_type") == "im"

    def is_need_response(self) -> bool:
        return self.classification in ("question",)

    def _is_mention_to_subteam(self) -> bool:
        return bool(re.search(r'<!subteam\^[^>]+>', self.text()))

    def is_mention_to_subteam(self) -> bool:
        return any([
            self._is_mention_to_subteam(),
            *[event._is_mention_to_subteam() for event in self.canceled_events]
        ])

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def id(self) -> str:
        return hashlib.sha256(f"{self.channel_id()},{self.thread_ts()},{self.ts()}".encode('utf8')).hexdigest()[:16]
