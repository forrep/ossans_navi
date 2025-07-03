import base64
import dataclasses
import datetime
import hashlib
import itertools
import re
from threading import RLock
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field


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
    bot_id: Optional[str] = dataclasses.field(default=None)


@dataclasses.dataclass
class SlackAttachment:
    title: str
    text: str
    link: str
    user: Optional[SlackUser] = dataclasses.field(default=None, init=False)

    @staticmethod
    def from_model(attachment: 'SlackAttachmentType') -> 'SlackAttachment':
        return SlackAttachment(
            title=attachment.title,
            text=attachment.text,
            link=attachment.title_link or attachment.from_url or "",
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

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class SlackFile:
    title: str
    mimetype: str
    filetype: str
    url_private: str
    pretty_type: str
    permalink: str
    is_public: bool
    is_initialized: bool = dataclasses.field(default=False, init=False)
    is_analyzed: bool = dataclasses.field(default=False, init=False)
    description: Optional[str] = dataclasses.field(default=None, init=False)
    text: Optional[str] = dataclasses.field(default=None, init=False)
    _content: Optional[bytes] = dataclasses.field(default=None, init=False)
    _height: int = dataclasses.field(default=0, init=False)
    _width: int = dataclasses.field(default=0, init=False)

    @staticmethod
    def from_model(file: 'SlackFileType') -> Optional['SlackFile']:
        if file.permalink is None or file.is_public is None or file.url_private is None:
            # 必須項目が1つでも存在しなければ None を返す（呼び出し元で None を除外する想定）
            return None
        slack_file = SlackFile(
            title=file.title,
            mimetype=file.mimetype,
            filetype=file.filetype,
            url_private=file.url_private,
            pretty_type=file.pretty_type,
            permalink=file.permalink,
            is_public=file.is_public,
        )
        if slack_file.is_text and file.plain_text:
            # text で plain_text があればそれを利用する
            slack_file.is_initialized = True
            slack_file.text = file.plain_text
        return slack_file

    @property
    def content(self) -> bytes:
        if self._content is None:
            raise ValueError("SlackFile's content is None.")
        return self._content

    @content.setter
    def content(self, value: bytes):
        self._content = value

    @property
    def is_image(self) -> bool:
        return self.mimetype.startswith("image/")

    @property
    def is_text(self) -> bool:
        return self.mimetype.startswith("text/")

    @property
    def is_canvas(self) -> bool:
        return (
            self.mimetype == "application/vnd.slack-docs"
            and self.filetype == "quip"
            and self.pretty_type == "canvas"
        )

    @property
    def is_textualize(self) -> bool:
        return self.is_text or self.is_canvas or (self.is_image and self.is_analyzed)

    @property
    def width(self) -> int:
        if not self.is_initialized:
            raise ValueError("SlackFile is not initialized.")
        return self._width

    @property
    def height(self) -> int:
        if not self.is_initialized:
            raise ValueError("SlackFile is not initialized.")
        return self._height

    @property
    def is_valid(self) -> bool:
        return self._content is not None

    @property
    def image_uri(self) -> str:
        return f"data:image/png;base64,{base64.b64encode(self.content).decode()}"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "mimetype": self.mimetype,
            "pretty_type": self.pretty_type,
            "url_private": self.url_private,
            "permalink": self.permalink,
            **(
                {
                    "width": self._width,
                    "height": self._height,
                } if self.is_image and self._width > 0 and self._height > 0 else {}
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

    def has_not_analyzed_files(self) -> bool:
        return len([file for file in self.files if file.is_image and not file.is_analyzed]) > 0

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
    is_initialized: bool = dataclasses.field(default=False, init=False)
    _root_message: Optional[SlackMessageLite] = dataclasses.field(default=None, init=False)
    _messages: list[SlackMessageLite] = dataclasses.field(default_factory=list, init=False)
    _is_full: bool = dataclasses.field(default=False, init=False)

    @property
    def root_message(self) -> SlackMessageLite | None:
        if not self.is_initialized:
            raise ValueError("SlackMessage is not initialized.")
        return self._root_message

    @property
    def messages(self) -> list[SlackMessageLite]:
        if not self.is_initialized:
            raise ValueError("SlackMessage is not initialized.")
        if self._messages is None:
            raise ValueError("SlackMessage's messages is None.")
        return self._messages

    @property
    def is_full(self) -> bool:
        if not self.is_initialized:
            raise ValueError("SlackMessage is not initialized.")
        return self._is_full

    @property
    def permalink(self) -> str:
        return self.message.permalink

    @property
    def files(self):
        if not self.is_initialized:
            raise ValueError("SlackMessage is not initialized.")
        return [
            *self.message.files,
            *(v.files if (v := self.root_message) else []),
            *(itertools.chain.from_iterable([v.files for v in self.messages])),
        ]

    @staticmethod
    def sort(messages: list['SlackMessage']) -> list['SlackMessage']:
        return sorted(messages, key=lambda v: v.message.timestamp, reverse=True)


@dataclasses.dataclass(order=True)
class SlackSearchTerm:
    _order_index: tuple[Any, ...] = dataclasses.field(init=False, repr=False)
    words: frozenset[str]
    date_from: Optional[datetime.datetime]
    date_to: Optional[datetime.datetime]

    def to_term(self, expand: bool = False) -> str:
        expand_days = 1 if expand else 0
        return (
            " ".join([
                *self.words,
                *([f"after:{(self.date_from + datetime.timedelta(days=-1 * expand_days)).strftime("%Y-%m-%d")}"] if self.date_from else []),
                *([f"before:{(self.date_to + datetime.timedelta(days=1 * expand_days)).strftime("%Y-%m-%d")}"] if self.date_to else []),
            ])
        )

    def __post_init__(self) -> None:
        # 並び替え専用項目（_order_index）を構築、以下の順に並べる
        #   1. ワード数が少ない
        #   2. ワードの合計文字数が少ない
        #   3. ワードのコード順
        #   4. 絞り込み期間が長い
        #   5. 絞り込み期間の開始がより過去
        #   6. 絞り込み期間の終了がより未来
        self._order_index = (
            len(self.words),
            sum([len(w) for w in self.words]),
            tuple(sorted(self.words)),
            (datetime.datetime.min if self.date_from is None else self.date_from) - (datetime.datetime.max if self.date_to is None else self.date_to),
            (datetime.datetime.min if self.date_from is None else self.date_from),
            datetime.datetime.min - (datetime.datetime.max if self.date_to is None else self.date_to),
        )

    def __hash__(self):
        return hash((self.words, self.date_from, self.date_to,))

    def __eq__(self, value):
        if not isinstance(value, SlackSearchTerm):
            return False
        return (
            self.words == value.words
            and self.date_from == value.date_from
            and self.date_to == value.date_to
        )

    def is_subset(self, other: "SlackSearchTerm") -> bool:
        return (
            other.words >= self.words
            and (
                (datetime.datetime.min if other.date_from is None else other.date_from)
                >= (datetime.datetime.min if self.date_from is None else self.date_from)
            )
            and (
                (datetime.datetime.max if other.date_to is None else other.date_to)
                <= (datetime.datetime.max if self.date_to is None else self.date_to)
            )
        )

    @staticmethod
    def parse(term: str) -> Optional["SlackSearchTerm"]:
        if re.search(r'\bfrom:(?!<@[A-Z0-9]+>)', term):
            # from:@XXX と指定するはずが from:<@名前> のような不正フォーマットが指定されると Slack API がエラーを返すので有効な絞り込み条件として採用しない
            # ※ LLM がたまに生成してしまう
            return None
        words: list[str] = []
        date_from: Optional[datetime.datetime] = None
        date_to: Optional[datetime.datetime] = None
        for word in term.split():
            if (matched := re.match(r"(before|after):([0-9]{4}-[0-9]{2}-[0-9]{2})", word)):
                if matched.group(1) == "before":
                    date_to = datetime.datetime.strptime(matched.group(2), '%Y-%m-%d')
                if matched.group(1) == "after":
                    date_from = datetime.datetime.strptime(matched.group(2), '%Y-%m-%d')
            else:
                words.append(word)
        return SlackSearchTerm(frozenset(words), date_from, date_to)


@dataclasses.dataclass
class SlackSearch:
    term: SlackSearchTerm
    total_count: int
    messages: list[SlackMessage]
    is_full: bool
    is_additional: bool
    is_get_messages: bool

    @property
    def words(self) -> str:
        return self.term.to_term()

    def get_id(self) -> str:
        return hashlib.md5(self.words.encode('utf8')).hexdigest()

    def is_meny_messages(self) -> bool:
        return len(self.messages) >= 40

    def is_too_meny_messages(self) -> bool:
        return len(self.messages) >= 60


@dataclasses.dataclass
class SlackSearches:
    results: list[SlackSearch] = dataclasses.field(default_factory=list, init=False)
    messages: dict[str, SlackMessage] = dataclasses.field(default_factory=dict, init=False)
    files: dict[str, SlackFile] = dataclasses.field(default_factory=dict, init=False)
    total_count: int = dataclasses.field(default=0, init=False)
    _used: set[str] = dataclasses.field(default_factory=set, init=False)
    _lastshot: dict[str, SlackMessage] = dataclasses.field(default_factory=dict, init=False)
    _lastshot_terms: set[str] = dataclasses.field(default_factory=set, init=False)
    _lock: RLock = dataclasses.field(default_factory=RLock, init=False)
    _lastshot_permalinks: set[str] = dataclasses.field(default_factory=set, init=False)

    def add(self, result: SlackSearch) -> None:
        # 同一 permalink の SlackMessage は 1つのインスタンスにまとめる
        result.messages = [self.messages.setdefault(message.permalink, message) for message in result.messages]
        self.results = sorted([*self.results, result], key=lambda v: v.total_count)
        self.total_count = sum([len(v.messages) for v in self.results])

    def __iter__(self):
        return iter(self.results)

    def result_len(self) -> int:
        return len(self.results)

    def use(self, permalinks: list[str] | str) -> None:
        if isinstance(permalinks, list):
            self._used.update(permalinks)
        if isinstance(permalinks, str):
            self._used.add(permalinks)

    def is_used(self, permalink: str) -> bool:
        return permalink in self._used

    def add_lastshot(self, permalinks: list[str]) -> None:
        """
        lastshot で RAG の結果として入力するメッセージのリストに追加する
        ただし、すでにメッセージが追加済みの場合は二重で追加しない
        渡された permalinks から検索結果内からメッセージを探す
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
                if not message.is_initialized:
                    # initialize() されていないメッセージはまだ処理対象外なので確認の必要はない
                    # そのメッセージが initialize() されたタイミングで確認される
                    continue
                if permalink == message.permalink:
                    # lastshot に追加したい message が見つかった
                    self._lastshot[message.permalink] = message
                    self._lastshot_terms.add(result.term.to_term())
                    self._lastshot_permalinks.add(message.permalink)
                    if message.is_full:
                        # そのメッセージにスレッド内の全メッセージが含まれている場合はそのスレッド内の全メッセージの permalink を 追加済みとしてマークする
                        # なぜならそのメッセージにはスレッド内の全メッセージが含まれているので、追加で別メッセージを読み込む必要がないため
                        self._lastshot_permalinks.update([v.permalink for v in message.messages])
                        if (v := message.root_message):
                            self._lastshot_permalinks.add(v.permalink)
                    return
        for result in self.results:
            # root_message とスレッド内のメッセージを探索する
            for message in result.messages:
                if not message.is_initialized:
                    # initialize() されていないメッセージはまだ処理対象外なので確認の必要はない
                    # そのメッセージが initialize() されたタイミングで確認される
                    continue
                if (
                    ((v := message.root_message) and permalink == v.permalink)
                    or (permalink in [v.permalink for v in message.messages])
                ):
                    # root_message またはスレッド内のメッセージの permalink と一致
                    self._lastshot[message.permalink] = message
                    self._lastshot_terms.add(result.term.to_term())
                    self._lastshot_permalinks.add(message.permalink)
                    if message.is_full:
                        # そのメッセージにスレッド内の全メッセージが含まれている場合はそのスレッド内の全メッセージの permalink を 追加済みとしてマークする
                        # なぜならそのメッセージにはスレッド内の全メッセージが含まれているので、追加で別メッセージを読み込む必要がないため
                        self._lastshot_permalinks.update([v.permalink for v in message.messages])
                        if (v := message.root_message):
                            self._lastshot_permalinks.add(v.permalink)
                    return

    @property
    def lastshot_messages(self) -> Iterable[SlackMessage]:
        return self._lastshot.values()

    @property
    def lastshot_terms(self) -> list[str]:
        return list(self._lastshot_terms)

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
    classification: Optional[dict[str, str | list[str]]] = dataclasses.field(init=False)
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

    @property
    def channel_id(self) -> str:
        return self.source["channel"]

    @property
    def text(self) -> str:
        if self.is_message_changed():
            # メッセージが変更された場合、変更後のメッセージを返す
            return self.source["message"]["text"]
        if self.is_message_deleted():
            # メッセージが削除された場合、固定メッセージを返す
            return "This message was deleted."
        return self.source["text"]

    @property
    def user_id(self) -> str:
        if self.is_message_changed():
            # メッセージが変更された場合、変更後のユーザー ID を返す
            return self.source["message"]["user"]
        if self.is_message_deleted():
            # メッセージが削除された場合、変更前のユーザー ID を返す（そもそも変更後のメッセージというデータが存在しないため）
            return self.source["previous_message"]["user"]
        return self.source["user"]

    @property
    def is_user(self) -> bool:
        if self.is_message_changed():
            # メッセージが変更された場合、変更後のユーザー ID を返す
            return "user" in self.source["message"]
        if self.is_message_deleted():
            # メッセージが削除された場合、変更前のユーザー ID を返す（そもそも変更後のメッセージというデータが存在しないため）
            return "user" in self.source["previous_message"]
        return "user" in self.source

    @property
    def bot_id(self) -> str:
        if self.is_message_changed() or self.is_message_deleted():
            # ボットによる更新・削除イベントは対応していない
            raise ValueError("Cannot get bot_id with message_changed or message_deleted")
        return self.source["bot_id"]

    @property
    def is_bot(self) -> bool:
        return not self.is_user

    @property
    def ts(self) -> str:
        if self.is_message_changed():
            # メッセージが変更された場合でも、元メッセージの送信日時（ts）を返す
            # ts は変更されないので self.source["previous_message"]["ts"] と self.source["message"]["ts"] は同じ内容のはずだがパターン網羅はできていない
            # self.source["ts"] には更新日時が保持されているが使わない
            return self.source["previous_message"]["ts"]
        if self.is_message_deleted():
            # メッセージが削除された場合でも、元メッセージの送信日時（ts）を返す
            # is_message_changed() と is_message_deleted() のパターンで分岐の必要は無いが、分かりやすさのため分岐している
            return self.source["previous_message"]["ts"]
        return self.source["ts"]

    @property
    def event_ts(self) -> str:
        """
        イベントのタイムスタンプを返却する、 event_ts という名称だが source["ts"] を返す
        通常イベントでは source["ts"] に送信日、更新イベントでは source["ts"] に更新日が保持されている
        SlackMessageEvent.ts は更新イベントでも元イベントの ts を返すため、常に最新の ts を返却する当メソッドとは仕様が異なる
        source["event_ts"] と source["ts"] は同じ値が保持されているようだが、正確な仕様が不明のため、ここでは source["ts"] を返却している
        なぜならば、更新・削除イベントで元イベントを探すために source["previous_message"]["ts"] と source["ts"] を比較するシーンで、
        source["event_ts"] を利用するよりも整合性が取れている
        """
        return self.source["ts"]

    @property
    def thread_ts(self) -> str:
        # スレッドの場合は親メッセージの ts を持つ thread_ts を返す
        # スレッドではない場合はそのメッセージからスレッドを作るので ts を返す
        if self.is_message_changed():
            # メッセージが変更された場合でも、元メッセージの thread_ts を返す
            # thread_ts は変更されないので self.source["previous_message"]["thread_ts"] と self.source["message"]["thread_ts"] は同じ内容のはずだがパターン網羅はできていない
            return (
                self.source["previous_message"]["thread_ts"]
                if "thread_ts" in self.source["previous_message"] else
                self.source["previous_message"]["ts"]
            )
        if self.is_message_deleted():
            # メッセージが削除された場合でも、元メッセージの thread_ts を返す
            # 削除には2パターンあり、スレッドのルートメッセージ削除された場合は message_changed イベントで論理削除される、この場合は source["message"] と source["previous_message"] の両方が存在する
            # しかし message_deleted イベントの場合は source["message"] が存在しないため、両方のパターンで使える source["previous_message"] を利用する
            # is_message_changed と is_message_deleted は同一の仕様だが、コメントなど分かりやすさのために処理を分けている
            return (
                self.source["previous_message"]["thread_ts"]
                if "thread_ts" in self.source["previous_message"] else
                self.source["previous_message"]["ts"]
            )
        return self.source["thread_ts"] if "thread_ts" in self.source else self.source["ts"]

    def _mentions(self) -> list[str]:
        return [user_id[0] for user_id in re.findall(r'<@([A-Z0-9]+)(\|[^>]+?)?>', self.text)]

    @property
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
        return bool(re.search(r'<!(?:channel|here)>', self.text))

    def is_broadcast(self) -> bool:
        return any([
            self._is_broadcast(),
            *[event._is_broadcast() for event in self.canceled_events]
        ])

    def is_reply_to_ossans_navi(self) -> bool:
        return not self.is_talk_to_other and self.is_next_message_from_ossans_navi

    def is_thread(self) -> bool:
        if self.is_message_changed() or self.is_message_deleted():
            # 変更イベント・削除イベントでは元メッセージを元にスレッド判定を行う
            # スレッドかどうかは変化しないので self.source["previous_message"] と self.source["message"] で同じ判定が可能なはずだがパターン網羅はできていない
            return "thread_ts" in self.source["previous_message"]
        return 'thread_ts' in self.source

    def is_message_post(self) -> bool:
        """
        応答が必要なメッセージ（1, 2 の条件に当てはまる）に True を返却
        1. type: message である
        2. text, channel, ts が存在する
        3. user, bot_id どちらかが存在する
        4. 次のいずれかに当てはまる
            - subtype が存在しない → 通常メッセージ
            - subtype が file_share → テキストスニペットの送信（ボットによるファイル送信も含む）
            - subtype が thread_broadcast → チャネルにも投稿するチェックを入れてスレッド返信
            - subtype が bot_message → ボットの通常メッセージ
        """
        return (
            self.source.get("type") == "message"
            and all(v in self.source for v in ("text", "channel", "ts", ))
            and any(v in self.source for v in ("user", "bot_id", ))
            and (
                "subtype" not in self.source
                or self.source["subtype"] in ("file_share", "thread_broadcast", "bot_message", )
            )
        )

    def is_message_changed(self) -> bool:
        """
        メッセージの編集イベントである場合は True を返却
        """
        if (
            self.source.get("type") == "message"
            and self.source.get("subtype") == "message_changed"
            and all(v in self.source for v in ("message", "previous_message", "channel", "ts", ))
        ):
            message: dict[str, dict | str] = self.source["message"]
            if (
                message.get("type") == "message"
                # 編集イベントは bot_id に対応しない、user のみ対応とする
                and all(v in message for v in ("text", "user", "ts", ))
                # 普通の更新イベント時は "hidden" パラメータが存在しない
                # hidden: True は更新イベントに見せかけてスレッドのルートメッセージを削除した場合に発生するイベント
                and message.get("hidden", False) is False
            ):
                return True
        return False

    def is_message_deleted(self) -> bool:
        """
        メッセージの削除イベントである場合は True を返却
        """
        message: dict[str, dict | str]
        if (
            self.source.get("type") == "message"
            and self.source.get("subtype") == "message_changed"
            and all(v in self.source for v in ("message", "previous_message", "channel", "ts", ))
        ):
            message = self.source["message"]
            if (
                message.get("type") == "message"
                # 削除イベントは bot_id に対応しない、user のみ対応とする
                and all(v in message for v in ("text", "user", "ts", ))
                # 普通の更新イベント時は "hidden" パラメータが存在しない
                # hidden: True は更新イベントに見せかけてスレッドのルートメッセージを削除した場合に発生するイベント
                and message.get("hidden", False) is True
            ):
                return True
        elif (
            self.source.get("type") == "message"
            and self.source.get("subtype") == "message_deleted"
            and all(v in self.source for v in ("previous_message", "channel", "ts", ))
        ):
            message = self.source["previous_message"]
            if (
                message.get("type") == "message"
                and all(v in message for v in ("text", "user", "ts"))
            ):
                return True
        return False

    def is_open_channel(self) -> bool:
        return self.source.get("channel_type") == "channel"

    def is_dm(self) -> bool:
        return self.source.get("channel_type") == "im"

    def is_need_response(self) -> bool:
        if self.user_intent is not None:
            if self.user_intentions_type in ("need_answers_to_questions",):
                return True
            if self.user_intentions_type in ("ask_someone_to_do_something",):
                if self.who_to_talk_to in ("to_someone_well_informed",):
                    return True
            if self.user_emotions in ("be_troubled",):
                if self.who_to_talk_to in ("to_someone_well_informed",):
                    return True
        return False

    @property
    def is_need_additional_information(self) -> bool:
        return not (
            len(self.required_knowledge_types) > 0
            and self.required_knowledge_types[0] == "no_information_required"
        )

    @property
    def user_intent(self) -> Optional[str]:
        return v if isinstance((v := (self.classification or {}).get("user_intent")), str) else None

    @property
    def user_intentions_type(self) -> str:
        return v if isinstance((v := (self.classification or {}).get("user_intentions_type")), str) else "no_intent"

    @property
    def who_to_talk_to(self) -> str:
        return v if isinstance((v := (self.classification or {}).get("who_to_talk_to")), str) else "cannot_determine"

    @property
    def user_emotions(self) -> str:
        return v if isinstance((v := (self.classification or {}).get("user_emotions")), str) else "no_emotions"

    @property
    def required_knowledge_types(self) -> list[str]:
        w = v if isinstance((v := (self.classification or {}).get("required_knowledge_types")), list) else []
        return [z for z in w if isinstance(z, str)]

    @property
    def slack_emoji_names(self) -> list[str]:
        return [
            *(v if isinstance(v := (self.classification or {}).get("slack_emoji_names"), list) else []),
            *(
                [v] if isinstance(
                    (
                        v := {
                            "need_answers_to_questions": None,                                # 質問
                            "ask_someone_to_do_something": None,                              # 依頼
                            "report_to_someone": "spiral_note_pad",                           # 報告
                            "advice_to_someone": "teacher",                                   # アドバイス
                            "agree_with_someone": "smile",                                    # 同意
                            "sympathize_with_someone": "relaxed",                             # 共感
                            "comfirm_with_someone": "wink",                                   # 確認
                            "praise_someone_or_something": "thumbsup",                        # 感嘆
                            "disappointed_in_someone_or_something": "disappointed_relieved",  # 失望
                            "sharing_information": "spiral_note_pad",                         # 共有
                            "note_for_self": "pencil",                                        # メモ
                            "other": None,
                        }.get(self.user_intentions_type)
                    ),
                    str
                ) else []
            )
        ]

    def _is_mention_to_subteam(self) -> bool:
        return bool(re.search(r'<!subteam\^[^>]+>', self.text))

    def is_mention_to_subteam(self) -> bool:
        return any([
            self._is_mention_to_subteam(),
            *[event._is_mention_to_subteam() for event in self.canceled_events]
        ])

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def id(self) -> str:
        return hashlib.sha256(f"{self.channel_id},{self.thread_ts},{self.ts}".encode('utf8')).hexdigest()[:16]


class SlackBaseResponse(BaseModel):
    """Slack API レスポンスの基本型定義"""
    ok: bool
    error: Optional[str] = None


class SlackAuthTestUserResponse(SlackBaseResponse):
    """Slack auth.test API レスポンスの型定義"""
    url: str
    team: str
    user: str
    team_id: str
    user_id: str


class SlackAuthTestBotResponse(SlackBaseResponse):
    """Slack auth.test API レスポンスの型定義"""
    url: str
    team: str
    user: str
    team_id: str
    user_id: str
    bot_id: str


class SlackAuthTestAppResponse(SlackBaseResponse):
    """Slack auth.test API レスポンスの型定義"""
    app_name: str
    app_id: str


class SlackUserType(BaseModel):
    """Slack users.info.user API レスポンスの型定義"""
    id: str
    name: str = "Unknown"
    real_name: str = "Unknown"
    profile: dict[str, Any] = Field(default_factory=dict)
    deleted: bool = False
    is_admin: bool = False
    is_restricted: bool = True
    is_bot: bool = False
    is_stranger: bool = False


class SlackUsersInfoResponse(SlackBaseResponse):
    """Slack users.info API レスポンスの型定義"""
    user: SlackUserType


class SlackBotResponse(BaseModel):
    """Slack bots.info.bot API レスポンスの型定義"""
    id: str
    user_id: Optional[str] = None
    deleted: bool = False
    name: str = "Unknown Bot"


class SlackBotsInfoResponse(SlackBaseResponse):
    """Slack bots.info API レスポンスの型定義"""
    bot: SlackBotResponse


class SlackChannelTopicPurposeType(BaseModel):
    """Slack channels.info.channel.topic/purpose API レスポンスの型定義"""
    value: str = ""


class SlackChannelType(BaseModel):
    """Slack channels.info.channel API レスポンスの型定義"""
    id: str
    name: str = ""
    topic: SlackChannelTopicPurposeType = Field(default_factory=SlackChannelTopicPurposeType)
    purpose: SlackChannelTopicPurposeType = Field(default_factory=SlackChannelTopicPurposeType)
    is_private: bool = False
    is_im: bool = False
    is_mpim: bool = False
    num_members: int = 0


class SlackConversationsInfoResponse(SlackBaseResponse):
    """Slack conversations.info API レスポンスの型定義"""
    channel: SlackChannelType


class SlackUsersGetPresenceResponse(SlackBaseResponse):
    """Slack users.getPresence API レスポンスの型定義"""
    presence: str

    @property
    def is_active(self) -> bool:
        """ユーザーがアクティブかどうかを判定する"""
        return self.presence == "active"


class SlackConversationsMembersResponse(SlackBaseResponse):
    """Slack conversations.members API レスポンスの型定義"""
    members: list[str] = Field(default_factory=list)


class SlackConversationsListResponse(SlackBaseResponse):
    """Slack conversations.list API レスポンスの型定義"""
    channels: list[SlackChannelType] = Field(default_factory=list)


class SlackFileType(BaseModel):
    id: str
    url_private: Optional[str] = None
    permalink: Optional[str] = None
    is_public: Optional[bool] = None
    title: str = ""
    mimetype: str = ""
    filetype: str = ""
    pretty_type: str = ""
    plain_text: Optional[str] = None


class SlackAttachmentType(BaseModel):
    """Slack conversations.replies/conversations.history API レスポンスのアタッチメント型定義"""
    name: str = ""
    title: str = ""
    text: str = ""
    author_id: str = ""
    from_url: Optional[str] = None
    title_link: Optional[str] = None

    @property
    def link(self) -> str:
        """アタッチメントのリンクを返す"""
        return self.title_link or self.from_url or ""


class SlackReactionType(BaseModel):
    """Slack reactions.add API レスポンスのリアクション型定義"""
    name: str
    count: int
    users: list[str] = Field(default_factory=list)
    is_custom_emoji: bool = False


class SlackMessageType(BaseModel):
    """Slack conversations.replies/conversations.history API レスポンスのメッセージ型定義"""
    ts: str
    text: str = ""
    channel: Optional[SlackChannelType] = None
    score: float = 0.0
    permalink: str = ""
    type: str = ""
    user: Optional[str] = None
    bot_id: Optional[str] = None
    attachments: list[SlackAttachmentType] = Field(default_factory=list)
    files: list[SlackFileType] = Field(default_factory=list)
    thread_ts: Optional[str] = None
    reactions: list[SlackReactionType] = Field(default_factory=list)


class SlackConversationsRepliesResponse(SlackBaseResponse):
    """Slack conversations.replies API レスポンスの型定義"""
    messages: list[SlackMessageType]


class SlackPaginationType(BaseModel):
    """Slack API レスポンスのページング情報型定義"""
    total_count: int = 0
    page: int = 1
    first: int = 1
    last: int = 0


class SlackSearchMessagesType(BaseModel):
    """Slack search.messages API レスポンスのメッセージ型定義"""
    total: int = 0
    pagination: SlackPaginationType = Field(default_factory=SlackPaginationType)
    matches: list[SlackMessageType] = Field(default_factory=list)


class SlackSearchMessagesResponse(SlackBaseResponse):
    """Slack search.messages API レスポンスの型定義"""
    query: str
    messages: SlackSearchMessagesType


class SlackConversationsOpenResponse(SlackBaseResponse):
    """Slack conversations.open API レスポンスの型定義"""
    channel: SlackChannelType


class SlackConversationsHistoryResponse(SlackBaseResponse):
    """Slack conversations.history API レスポンスの型定義"""
    messages: list[SlackMessageType] = Field(default_factory=list)
