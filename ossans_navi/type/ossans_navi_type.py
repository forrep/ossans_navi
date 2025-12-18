from typing import Any, Iterable

from pydantic import BaseModel, Field, PrivateAttr

from ossans_navi.type.slack_type import SlackFile, SlackMessage, SlackSearch


class Image(BaseModel):
    data: bytes
    mime_type: str

    @property
    def extension(self) -> str:
        if self.mime_type == "image/png":
            return ".png"
        elif self.mime_type == "image/jpeg":
            return ".jpg"
        elif self.mime_type == "image/gif":
            return ".gif"
        else:
            return ""


class OssansNaviConfig(BaseModel):
    trusted_bots: list[str] = Field(default_factory=list, init=False)
    allow_responds: list[str] = Field(default_factory=list, init=False)
    admin_users: list[str] = Field(default_factory=list, init=False)
    viewable_private_channels: list[str] = Field(default_factory=list, init=False)

    @classmethod
    def from_dict(cls, settings_dict: dict[str, Any]) -> 'OssansNaviConfig':
        settings = cls()
        if settings_dict.get("type") == "config":
            if isinstance(settings_dict.get("trusted_bots"), list):
                settings.trusted_bots.extend([v for v in settings_dict["trusted_bots"] if isinstance(v, str) and v.startswith("U")])
            if isinstance(settings_dict.get("allow_responds"), list):
                settings.allow_responds.extend([v for v in settings_dict["allow_responds"] if isinstance(v, str) and v.startswith("U")])
            if isinstance(settings_dict.get("admin_users"), list):
                settings.admin_users.extend([v for v in settings_dict["admin_users"] if isinstance(v, str) and v.startswith("U")])
            if isinstance(settings_dict.get("viewable_private_channels"), list):
                settings.viewable_private_channels.extend([v for v in settings_dict["viewable_private_channels"] if isinstance(v, str)])
        return settings

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class LastshotResponse(BaseModel):
    text: str
    images: list[Image]


class UrlContext(BaseModel):
    url: str
    content: str


class SearchResults(BaseModel):
    slack_search_results: list[SlackSearch] = Field(default_factory=list, init=False)
    url_context_results: list[UrlContext] = Field(default_factory=list, init=False)
    messages: dict[str, SlackMessage] = Field(default_factory=dict, init=False)
    files: dict[str, SlackFile] = Field(default_factory=dict, init=False)
    slack_search_messages_len: int = Field(default=0, init=False)
    _used: set[str] = PrivateAttr(default_factory=set, init=False)
    _lastshot: dict[str, SlackMessage] = PrivateAttr(default_factory=dict, init=False)
    _lastshot_terms: set[str] = PrivateAttr(default_factory=set, init=False)
    _lastshot_permalinks: set[str] = PrivateAttr(default_factory=set, init=False)

    def add(self, result: SlackSearch | UrlContext | list[SlackSearch | UrlContext]) -> None:
        if isinstance(result, list):
            for v in result:
                self.add(v)
        if isinstance(result, SlackSearch):
            # 同一 permalink の SlackMessage は 1つのインスタンスにまとめる
            result.messages = [self.messages.setdefault(message.permalink, message) for message in result.messages]
            self.slack_search_results = sorted([*self.slack_search_results, result], key=lambda v: v.total_count)
            self.slack_search_messages_len = sum([len(v.messages) for v in self.slack_search_results])
        if isinstance(result, UrlContext):
            self.url_context_results.append(result)

    @property
    def url_context_urls(self) -> set[str]:
        return {v.url for v in self.url_context_results}

    @property
    def slack_search_results_len(self) -> int:
        return len(self.slack_search_results)

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
        for result in self.slack_search_results:
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
        for result in self.slack_search_results:
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
