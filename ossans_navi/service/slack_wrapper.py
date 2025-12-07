import asyncio
import functools
import logging
import os
from io import IOBase
from typing import Any, Awaitable, Callable, Dict, List, Optional, ParamSpec, Sequence, TypeVar, Union

from slack_sdk.errors import SlackApiError
from slack_sdk.models.attachments import Attachment
from slack_sdk.models.blocks import Block
from slack_sdk.models.metadata import Metadata
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.web.async_slack_response import AsyncSlackResponse
from slack_sdk.web.slack_response import SlackResponse

T = TypeVar('T')
P = ParamSpec('P')

logger = logging.getLogger(__name__)


def api_wrapper(
    *,
    pager: Optional[Callable[[AsyncSlackResponse, dict], bool]] = None,
    concat: Optional[Callable[[AsyncSlackResponse, AsyncSlackResponse], AsyncSlackResponse]] = None,
    limit: int = 100,
    concurrency: int = 2,
) -> Callable[[Callable[P, Awaitable[AsyncSlackResponse]]], Callable[P, Awaitable[AsyncSlackResponse]]]:
    semaphore = asyncio.Semaphore(concurrency)

    def _api_wrapper(func: Callable[P, Awaitable[AsyncSlackResponse]]) -> Callable[P, Awaitable[AsyncSlackResponse]]:
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs) -> AsyncSlackResponse:
            response: Optional[AsyncSlackResponse] = None
            page = 0
            while limit > page:
                for _ in range(20):
                    try:
                        async with semaphore:
                            # 各APIメソッドごとに並列数を制限する、Slack 側の制限をなるべく超えないために
                            current_response = await func(*args, **kwargs)
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
                        logger.warning(e)
                        await asyncio.sleep(20)
            if response is None:
                raise RuntimeError('Response is None.')
            return response
        return _wrapper  # type: ignore
    return _api_wrapper


def cursor_pager(response: AsyncSlackResponse, kwargs: dict) -> bool:
    response_metadata: dict[str, str] = response.get("response_metadata", {})
    if len(next_cursor := response_metadata.get("next_cursor", "")) == 0:
        return True
    else:
        kwargs["cursor"] = next_cursor
        return False


def concat_response(name: str) -> Callable[[AsyncSlackResponse, AsyncSlackResponse], AsyncSlackResponse]:
    def _concat_response(response1: AsyncSlackResponse, response2: AsyncSlackResponse) -> AsyncSlackResponse:
        if (
            isinstance((v1 := response1.get(name)), list)
            and isinstance((v2 := response2.get(name)), list)
        ):
            v1.extend(v2)
        return response1
    return _concat_response


class SlackWrapper:
    def __init__(self, token: str) -> None:
        self.token = token
        self.client = AsyncWebClient(token=token)

    @api_wrapper()
    async def auth_test(self) -> AsyncSlackResponse:
        return await self.client.auth_test()

    @api_wrapper()
    async def users_info(
        self,
        *,
        user: str,
        include_locale: Optional[bool] = None,
    ) -> AsyncSlackResponse:
        return await self.client.users_info(user=user, include_locale=include_locale)

    @api_wrapper()
    async def bots_info(
        self,
        *,
        bot: Optional[str] = None,
        team_id: Optional[str] = None
    ) -> AsyncSlackResponse:
        return await self.client.bots_info(bot=bot, team_id=team_id)

    @api_wrapper()
    async def conversations_history(
        self,
        *,
        channel: str,
        cursor: Optional[str] = None,
        inclusive: Optional[bool] = None,
        include_all_metadata: Optional[bool] = None,
        latest: Optional[str] = None,
        limit: Optional[int] = None,
        oldest: Optional[str] = None,
    ) -> AsyncSlackResponse:
        return await self.client.conversations_history(
            channel=channel,
            cursor=cursor,
            inclusive=inclusive,
            include_all_metadata=include_all_metadata,
            latest=latest,
            limit=limit,
            oldest=oldest,
        )

    @api_wrapper(pager=cursor_pager, concat=concat_response(name="channels"), limit=999)
    async def conversations_list(
        self,
        *,
        cursor: Optional[str] = None,
        exclude_archived: Optional[bool] = None,
        limit: Optional[int] = None,
        team_id: Optional[str] = None,
        types: Optional[Union[str, Sequence[str]]] = None,
    ) -> AsyncSlackResponse:
        return await self.client.conversations_list(cursor=cursor, exclude_archived=exclude_archived, limit=limit, team_id=team_id, types=types)

    @api_wrapper()
    async def conversations_info(
        self,
        *,
        channel: str,
        include_locale: Optional[bool] = None,
        include_num_members: Optional[bool] = None,
    ) -> AsyncSlackResponse:
        return await self.client.conversations_info(
            channel=channel,
            include_locale=include_locale,
            include_num_members=include_num_members,
        )

    @api_wrapper(pager=cursor_pager, concat=concat_response(name="members"), limit=999)
    async def conversations_members(
        self,
        *,
        channel: str,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> AsyncSlackResponse:
        return await self.client.conversations_members(channel=channel, cursor=cursor, limit=limit)

    @api_wrapper()
    async def conversations_replies(
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
    ) -> AsyncSlackResponse:
        return await self.client.conversations_replies(
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
    async def users_getPresence(self, *, user: str) -> AsyncSlackResponse:
        return await self.client.users_getPresence(user=user)

    @api_wrapper()
    async def search_messages(
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
    ) -> AsyncSlackResponse:
        return await self.client.search_messages(
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
    async def reactions_add(
        self,
        *,
        channel: str,
        name: str,
        timestamp: str,
    ) -> AsyncSlackResponse:
        return await self.client.reactions_add(
            channel=channel,
            name=name,
            timestamp=timestamp,
        )

    @api_wrapper()
    async def reactions_remove(
        self,
        *,
        channel: str,
        name: str,
        timestamp: str,
    ) -> AsyncSlackResponse:
        return await self.client.reactions_remove(
            channel=channel,
            name=name,
            timestamp=timestamp,
        )

    @api_wrapper()
    async def chat_postMessage(
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
    ) -> AsyncSlackResponse:
        return await self.client.chat_postMessage(
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
    async def conversations_open(
        self,
        *,
        channel: Optional[str] = None,
        return_im: Optional[bool] = None,
        users: Optional[Union[str, Sequence[str]]] = None,
    ) -> AsyncSlackResponse:
        return await self.client.conversations_open(
            channel=channel,
            return_im=return_im,
            users=users,
        )

    @api_wrapper()
    async def files_upload_v2(
        self,
        *,
        # for sending a single file
        filename: Optional[str] = None,  # you can skip this only when sending along with content parameter
        file: Optional[Union[str, bytes, IOBase, os.PathLike]] = None,
        content: Optional[Union[str, bytes]] = None,
        title: Optional[str] = None,
        alt_txt: Optional[str] = None,
        snippet_type: Optional[str] = None,
        # To upload multiple files at a time
        file_uploads: Optional[List[Dict[str, Any]]] = None,
        channel: Optional[str] = None,
        channels: Optional[List[str]] = None,
        initial_comment: Optional[str] = None,
        thread_ts: Optional[str] = None,
        request_file_info: bool = True,  # since v3.23, this flag is no longer necessary
    ) -> AsyncSlackResponse:
        return await self.client.files_upload_v2(
            filename=filename,
            file=file,
            content=content,
            title=title,
            alt_txt=alt_txt,
            snippet_type=snippet_type,
            file_uploads=file_uploads,
            channel=channel,
            channels=channels,
            initial_comment=initial_comment,
            thread_ts=thread_ts,
            request_file_info=request_file_info,
        )

    @api_wrapper(pager=cursor_pager, concat=concat_response(name="members"), limit=999)
    async def users_list(
        self,
        *,
        cursor: Optional[str] = None,
        include_locale: Optional[bool] = None,
        limit: Optional[int] = None,
        team_id: Optional[str] = None,
    ) -> AsyncSlackResponse:
        return await self.client.users_list(
            cursor=cursor,
            include_locale=include_locale,
            limit=limit,
            team_id=team_id,
        )
