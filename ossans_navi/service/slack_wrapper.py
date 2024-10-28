import functools
import logging
import time
from threading import RLock
from typing import Callable, Dict, Optional, Sequence, TypeVar, Union

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.models.attachments import Attachment
from slack_sdk.models.blocks import Block
from slack_sdk.models.metadata import Metadata
from slack_sdk.web.base_client import SlackResponse

F = TypeVar('F', bound=Callable[..., SlackResponse])

logger = logging.getLogger(__name__)


def api_wrapper(
    *,
    pager: Optional[Callable[[SlackResponse, dict], bool]] = None,
    concat: Optional[Callable[[SlackResponse, SlackResponse], SlackResponse]] = None,
    limit: int = 100
) -> Callable[[F], F]:
    lock = RLock()

    def _api_wrapper(f: F) -> F:
        @functools.wraps(f)
        def _wrapper(*args, **kwargs) -> SlackResponse:
            response: Optional[SlackResponse] = None
            page = 0
            while limit > page:
                for _ in range(20):
                    try:
                        with lock:
                            # 各メソッドは1並列に制限してAPIコールする、多重で呼び出すと Slack 側に負荷がかかるため
                            current_response = f(*args, **kwargs)
                        if concat and response:
                            response = concat(response, current_response)
                        else:
                            response = current_response
                        if not pager or not concat:
                            page = limit
                        else:
                            page += 1
                            if pager(current_response, kwargs):
                                # pager の戻りが True なら、次のページがないということ
                                page = limit
                        # 例外が発生せずに実行できたら終了
                        break
                    except SlackApiError as e:
                        error_response: SlackResponse = e.response
                        if error_response.get("ok") or error_response.get("error") != "ratelimited":
                            # ok: true だったり、error: ratelimited ではない場合は例外をそのまま投げる
                            raise e
                        logger.warn(e)
                        time.sleep(20)
            if response is None:
                raise RuntimeError('Response is None.')
            return response
        return _wrapper  # type: ignore
    return _api_wrapper


def cursor_pager(response: SlackResponse, kwargs: dict) -> bool:
    response_metadata: dict[str, str] = response.get("response_metadata", {})
    if len(next_cursor := response_metadata.get("next_cursor", "")) == 0:
        return True
    else:
        kwargs["cursor"] = next_cursor
        return False


def concat_response(name: str) -> Callable[[SlackResponse, SlackResponse], SlackResponse]:
    def _concat_response(response1: SlackResponse, response2: SlackResponse) -> SlackResponse:
        if name not in response1 or name not in response2:
            return response1
        concat_list: list = response1[name]
        concat_list.extend(response2[name])
        return response1
    return _concat_response


class SlackWrapper:
    def __init__(self, token: str) -> None:
        self.token = token
        self.client = WebClient(token=token)

    @api_wrapper()
    def auth_test(self) -> SlackResponse:
        return self.client.auth_test()

    @api_wrapper()
    def users_info(
        self,
        *,
        user: str,
        include_locale: Optional[bool] = None,
    ) -> SlackResponse:
        return self.client.users_info(user=user, include_locale=include_locale)

    @api_wrapper()
    def bots_info(
        self,
        *,
        bot: Optional[str] = None,
        team_id: Optional[str] = None
    ) -> SlackResponse:
        return self.client.bots_info(bot=bot, team_id=team_id)

    @api_wrapper()
    def conversations_history(
        self,
        *,
        channel: str,
        cursor: Optional[str] = None,
        inclusive: Optional[bool] = None,
        include_all_metadata: Optional[bool] = None,
        latest: Optional[str] = None,
        limit: Optional[int] = None,
        oldest: Optional[str] = None,
    ) -> SlackResponse:
        return self.client.conversations_history(
            channel=channel,
            cursor=cursor,
            inclusive=inclusive,
            include_all_metadata=include_all_metadata,
            latest=latest,
            limit=limit,
            oldest=oldest,
        )

    @api_wrapper(pager=cursor_pager, concat=concat_response(name="channels"), limit=999)
    def conversations_list(
        self,
        *,
        cursor: Optional[str] = None,
        exclude_archived: Optional[bool] = None,
        limit: Optional[int] = None,
        team_id: Optional[str] = None,
        types: Optional[Union[str, Sequence[str]]] = None,
    ) -> SlackResponse:
        return self.client.conversations_list(cursor=cursor, exclude_archived=exclude_archived, limit=limit, team_id=team_id, types=types)

    @api_wrapper()
    def conversations_info(
        self,
        *,
        channel: str,
        include_locale: Optional[bool] = None,
        include_num_members: Optional[bool] = None,
    ) -> SlackResponse:
        return self.client.conversations_info(
            channel=channel,
            include_locale=include_locale,
            include_num_members=include_num_members,
        )

    @api_wrapper(pager=cursor_pager, concat=concat_response(name="members"), limit=999)
    def conversations_members(
        self,
        *,
        channel: str,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> SlackResponse:
        return self.client.conversations_members(channel=channel, cursor=cursor, limit=limit)

    @api_wrapper()
    def conversations_replies(
        self,
        *,
        channel: str,
        ts: str,
        cursor: Optional[str] = None,
        inclusive: Optional[bool] = None,
        include_all_metadata: Optional[bool] = None,
        latest: Optional[str] = None,
        limit: Optional[int] = None,
        oldest: Optional[str] = None
    ) -> SlackResponse:
        return self.client.conversations_replies(
            channel=channel,
            ts=ts,
            cursor=cursor,
            inclusive=inclusive,
            include_all_metadata=include_all_metadata,
            latest=latest,
            limit=limit,
            oldest=oldest,
        )

    @api_wrapper()
    def users_getPresence(self, *, user: str) -> SlackResponse:
        return self.client.users_getPresence(user=user)

    @api_wrapper()
    def search_messages(
        self,
        *,
        query: str,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
        highlight: Optional[bool] = None,
        page: Optional[int] = None,
        sort: Optional[str] = None,
        sort_dir: Optional[str] = None,
        team_id: Optional[str] = None
    ) -> SlackResponse:
        return self.client.search_messages(
            query=query,
            count=count,
            cursor=cursor,
            highlight=highlight,
            page=page,
            sort=sort,
            sort_dir=sort_dir,
            team_id=team_id
        )

    @api_wrapper()
    def reactions_add(
        self,
        *,
        channel: str,
        name: str,
        timestamp: str,
    ) -> SlackResponse:
        return self.client.reactions_add(
            channel=channel,
            name=name,
            timestamp=timestamp,
        )

    @api_wrapper()
    def chat_postMessage(
        self,
        *,
        channel: str,
        text: Optional[str] = None,
        as_user: Optional[bool] = None,
        attachments: Optional[Union[str, Sequence[Union[Dict, Attachment]]]] = None,
        blocks: Optional[Union[str, Sequence[Union[Dict, Block]]]] = None,
        thread_ts: Optional[str] = None,
        reply_broadcast: Optional[bool] = None,
        unfurl_links: Optional[bool] = None,
        unfurl_media: Optional[bool] = None,
        container_id: Optional[str] = None,
        icon_emoji: Optional[str] = None,
        icon_url: Optional[str] = None,
        mrkdwn: Optional[bool] = None,
        link_names: Optional[bool] = None,
        username: Optional[str] = None,
        parse: Optional[str] = None,  # none, full
        metadata: Optional[Union[Dict, Metadata]] = None
    ) -> SlackResponse:
        return self.client.chat_postMessage(
            channel=channel,
            text=text,
            as_user=as_user,
            attachments=attachments,
            blocks=blocks,
            thread_ts=thread_ts,
            reply_broadcast=reply_broadcast,
            unfurl_links=unfurl_links,
            unfurl_media=unfurl_media,
            container_id=container_id,
            icon_emoji=icon_emoji,
            icon_url=icon_url,
            mrkdwn=mrkdwn,
            link_names=link_names,
            username=username,
            parse=parse,
            metadata=metadata
        )

    @api_wrapper()
    def conversations_open(
        self,
        *,
        channel: Optional[str] = None,
        return_im: Optional[bool] = None,
        users: Optional[Union[str, Sequence[str]]] = None,
    ) -> SlackResponse:
        return self.client.conversations_open(
            channel=channel,
            return_im=return_im,
            users=users,
        )
