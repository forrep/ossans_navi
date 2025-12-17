import asyncio
import json
import logging
import re
import signal
import sys
from collections.abc import AsyncGenerator
from typing import Any, Callable, Coroutine

from slack_sdk import WebClient

from ossans_navi import config
from ossans_navi.common import async_utils
from ossans_navi.controller import config_controller
from ossans_navi.service.ai_service import AiService
from ossans_navi.service.ossans_navi_service import OssansNaviService
from ossans_navi.service.slack_service import EventGuard, SlackService
from ossans_navi.type.slack_type import SlackMessageEvent

EVENT_GUARD = EventGuard()
SEMAPHORE = asyncio.Semaphore(2)

slack_service = SlackService()

logger = logging.getLogger(__name__)


@slack_service.app.action(re.compile(r".*"))
async def handle_button_click(ack: Callable, body: dict[Any, Any], client: WebClient):
    await ack()
    logger.info(f"{json.dumps(body, ensure_ascii=False)}")
    await config_controller.routing(body, slack_service)


@slack_service.app.event("message")
async def handle_message_events(ack: Callable[[], Coroutine[Any, Any, None]], event: dict[str, dict]):
    await ack()
    logger.info("event=" + json.dumps(event, ensure_ascii=False))
    message_event = SlackMessageEvent(source=event)

    # 処理キュー(EVENT_GUARD)を操作する、途中に他の非同期処理を挟まないようにする
    if EVENT_GUARD.is_duplicate(message_event):
        logger.info("Duplicate event detected, finished.")
        return
    if message_event.is_message_post():
        # メッセージの投稿イベントの場合
        EVENT_GUARD.queue(message_event)
        logger.info(f"Event queued(message_post): {message_event.id()} ({message_event.id_source})")
    elif message_event.is_message_changed():
        # 更新イベントの場合
        if EVENT_GUARD.is_queueed_or_running(message_event):
            # 更新イベントは、元メッセージが QUEUED か RUNNING の場合（つまりまだ応答してない）場合に限ってキューに追加する
            # このパターンのためにロックが必要、ロックしてないと is_queueed_or_running 判定後に応答が終了する可能性もある
            EVENT_GUARD.queue(message_event)
            logger.info(
                f"Event queued(message_changed): {message_event.id()} ({message_event.id_source})"
            )
        else:
            logger.info("This message_changed event is terminated because it is not QUEUED or RUNNING.")
            return
    elif message_event.is_message_deleted():
        # 削除イベントの場合
        if EVENT_GUARD.is_queueed_or_running(message_event):
            # 削除された場合は、元メッセージに応答する必要がなくなる
            # 削除イベントは、該当メッセージが QUEUED か RUNNING の場合（つまりまだ応答してない）場合、そのキューをキャンセルする
            EVENT_GUARD.cancel(message_event)
            return
        else:
            logger.info("This message_deleted event is terminated because it is not QUEUED or RUNNING.")
            return
    else:
        logger.info("This is not message_post or message_changed event, finished")
        return
    # ここまで他の非同期処理を挟まないようにする

    async with SEMAPHORE:
        await do_ossans_navi_response_safe(message_event)

    logger.info(f"Event finished: {message_event.id()} ({message_event.id_source})")
    # 応答まで至ったパターンはロックを取ってから finish が必要なのですでに終了済み、重複して終了しても問題なし
    # それ以外パターンはまだ finish してないのでここで finish が必要
    EVENT_GUARD.finish(message_event)


async def do_ossans_navi_response_safe(event: SlackMessageEvent) -> None:
    # 処理キュー(EVENT_GUARD)を操作する、途中に他の非同期処理を挟まないようにする
    if EVENT_GUARD.is_canceled(event):
        logger.info(f"Event canceled: {event.id()} ({event.id_source})")
        logger.info("Finished.")
        return
    logger.info(f"Event started: {event.id()} ({event.id_source})")
    EVENT_GUARD.start(event)
    # ここまで他の非同期処理を挟まないようにする

    try:
        # event の構築作業
        event.canceled_events.extend(EVENT_GUARD.get_canceled_events(event))
        (event.user, event.channel) = await asyncio.gather(
            slack_service.get_user(event.user_id if event.is_user else event.bot_id),
            slack_service.get_channel(event.channel_id),
        )
        mentions = await async_utils.asyncio_gather(*[slack_service.get_user(user_id) for user_id in event.mentions], concurrency=3)
        if event.is_dm() or len([user for user in mentions if user.user_id in slack_service.my_bot_user_id]) > 0:
            # DM でのやりとり、またはメンションされている場合を「メンション」と扱う
            event.is_mention = True
        elif len(mentions) > 0 or event.is_mention_to_subteam() or event.is_broadcast():
            # OssansNavi にメンションしていない状態で、かつ他の人へのメンションや @here @channel がある場合。明らかに OssansNavi へ話かけていない
            event.is_talk_to_other = True
        logger.info(
            f"user={event.user.name}({event.user.user_id}), channel={event.channel.name}({event.channel.channel_id}), "
            + f"is_mention={event.is_mention}, is_dm={event.is_dm()}, is_talk_to_other={event.is_talk_to_other}"
        )

        ossans_navi_service = await OssansNaviService.create(
            slack_service=slack_service,
            event=event,
        )

        async for _ in do_ossans_navi_response(ossans_navi_service, event):
            # yield のタイミングで処理が戻されるごとにイベントがキャンセルされていないか確認して、キャンセルされていれば終了する
            if EVENT_GUARD.is_canceled(event):
                logger.info(f"Event canceled: {event.id()}({event.id_source})")
                logger.info("Finished.")
                break
        # キャンセルなどのパターンで終了した場合は、ここでリアクションを削除する
        await ossans_navi_service.remove_progress_reaction()
    except Exception as e:
        logger.error("do_ossans_navi_response return error")
        logger.error(e, exc_info=True)
        if event.is_mention:
            try:
                await slack_service.chat_post_message(
                    channel=event.channel_id,
                    thread_ts=event.thread_ts,
                    text=f"```{str(e)}```",
                )
            except Exception as e:
                logger.error(e, exc_info=True)
    finally:
        if ossans_navi_service:
            logger.info(f"Usage Report (Total Cost: {ossans_navi_service.ai_service.models_usage.get_total_cost():.4f})")
            for model in [v for v in ossans_navi_service.ai_service.models_usage.models if v.get_total_cost() > 0.0]:
                logger.info(f"  {model.model.name}({model.model_name}) Cost: {model.get_total_cost():.4f}")
                logger.info(f"    tokens_in  = {model.tokens_in}")
                logger.info(f"    tokens_out = {model.tokens_out}")


async def do_ossans_navi_response(
    ossans_navi_service: OssansNaviService,
    event: SlackMessageEvent,
) -> AsyncGenerator[None, None]:
    if event.is_dm() and await ossans_navi_service.special_command():
        # DM の場合は special_command() を実行、そして True が返ってきたら special_command を実行しているので通常のメッセージ処理は終了する
        return

    if event.is_mention:
        # 応答が確定している場合は処理中を示すリアクションを付けて UX を向上させる
        await ossans_navi_service.do_progress_reaction(config.PROGRESS_REACTION_THINKING)

    # yield で呼び出し元に戻すとイベントのキャンセルチェックをして、キャンセルされていれば終了する
    yield

    if event.is_talk_to_other:
        logger.info("Talk to other, finished.")
        return

    if not event.user:
        logger.info("User not found, finished")
        return

    if event.user.is_bot or event.user.is_guest:
        # ボットやゲストには基本的には応答しないが、特別な条件にヒットしたら応答可能、その判定を行う
        if (
            event.user.user_id in ossans_navi_service.config.allow_responds
            or (event.user.bot_id is not None and event.user.bot_id in ossans_navi_service.config.allow_responds)
        ):
            # 特別に許可されたユーザーに一致、この場合は応答が許可される
            logger.info(
                f"user: {event.user.user_id}{"(" + event.user.bot_id + ")" if event.user.bot_id is not None else ""} is in allow_responds list."
            )
        else:
            # OssansNavi はパブリックチャネルを無制限に検索して情報源にするため、パブリックチャネルに対する読み取り制限があるユーザーは利用できない
            logger.info("sending_user is bot or guest, finished.")
            return

    # production で起動している場合は開発用チャネルには応答しない
    if not config.DEVELOPMENT_MODE and event.channel_id in (config.DEVELOPMENT_CHANNELS):
        logger.info(f"Now in production mode. Ignoring development channel: {event.channel_id}")
        return

    # development で起動、DMで、かつ developer に入ってないなら応答しない
    # development で起動、非DMで、かつ非開発用チャネルには応答しない
    if config.DEVELOPMENT_MODE:
        if event.is_dm():
            if event.user.user_id not in config.DEVELOPERS:
                logger.info(f"Now in development mode. Ignoring non developer: {event.user.name}")
                return
        else:
            if event.channel_id not in (config.DEVELOPMENT_CHANNELS):
                logger.info(f"Now in development mode. Ignoring non development channel: {event.channel_id}")
                return

    # DM の topic から設定を取得する
    if event.is_user and not event.user.is_bot:
        event.settings = await slack_service.get_dm_info_with_ossans_navi(event.user.user_id)

    # スレッドでやりとりされた履歴メッセージを取得
    thread_messages = await ossans_navi_service.get_thread_messages()
    # スレッドの会話内に OssansNavi からの返信があるかどうか？（そのスレッドに参加しているかどうか）
    event.is_joined = ossans_navi_service.is_joined(thread_messages)
    # スレッド上の会話で、OssansNavi のメッセージの直後に送信されたメッセージか？（OssansNavi のメッセージへの返信だと思われるか）
    event.is_next_message_from_ossans_navi = ossans_navi_service.is_next_message_from_ossans_navi(thread_messages)
    logger.info(
        f"is_joined={event.is_joined},"
        + f" is_next_message_from_ossans_navi={event.is_next_message_from_ossans_navi},"
        + f" is_reply_to_ossans_navi={event.is_reply_to_ossans_navi()}"
    )

    if config.SILENT_MODE and not event.is_dm() and not event.is_mention and not event.is_reply_to_ossans_navi():
        # config.silent_mode が有効な場合は DM もしくはメンションされた場合、もしくは ossans_navi のメッセージの次のメッセージの場合のみ稼働する
        logger.info("OSSANS_NAVI_SILENT_MODE is enabled. Only 'mentions' or 'DMs' or 'reply to ossans_navi' will be responded to, finished.")
        return

    if (
        # スレッド内の投稿は以下のケースにマッチする場合は応答しない
        # - キャンセルしたイベントを含めて全てがスレッド内のやりとりである
        # - DMではない
        # - スレッドの会話に参加してない
        # - メンションされていない
        len([True for e in [event, *event.canceled_events] if not e.is_thread()]) == 0
        and not event.is_dm()
        and not event.is_joined
        and not event.is_mention
    ):
        logger.info("Message in threads where ossans_navi is not part of the conversation, finished.")
        return

    # 定期的にイベントがキャンセルされていないか確認して、キャンセルされていれば終了する
    yield

    # メッセージ内に後のフェーズで解析予定の画像・映像・音声ファイルがあるかどうかを判定する
    # 画像・映像・音声は文字列情報と違い、各処理フェーズで毎回は入力しない、必要なタイミングでのみ入力する仕様
    # 画像・映像・音声の解析（入力）タイミングについて
    #   - 画像: classify（メッセージの仕分け）の次フェーズで解析して文字列化する、設定によっては lastshot でも再入力する
    #   - 映像・音声: LOAD_VIDEO_AUDIO_FILES が有効な場合は lastshot で入力する、LOAD_VIDEO_AUDIO_FILES が無効な場合は入力しない、vtt は常に入力する
    # 例えば「添付した音声を要約して」という依頼に対して classify や do_slack_searches で音声を入力していないと検索によって無理に探そうとしてしまう
    # そのため「音声は後のフェーズで入力するから入力された前提で応答して」というシステムプロンプトを追加する必要がある
    async with asyncio.TaskGroup() as tg:
        for (i, message) in enumerate(thread_messages):
            is_latest_message = i == len(thread_messages) - 1
            for file in message.files:
                if file.is_image:
                    # 画像が存在する場合は常に解析対象となる
                    event.has_image_video_audio = True
                elif file.is_video or file.is_audio:
                    # 映像・音声は条件付きで lastshot に入力する
                    # vtt が存在する場合は、このタイミングで読み込む、classify が vtt 情報を参照できる
                    # タスクグループで一斉に読み込む
                    tg.create_task(ossans_navi_service.load_slack_file(file, user_client=False, load_file=False, load_vtt=True, initialized=False))
                    if is_latest_message and event.is_mention and config.LOAD_VIDEO_AUDIO_FILES:
                        # lastshot で入力する条件が上記の通り、その場合は「後で入力するよ」フラグを立てる
                        event.has_image_video_audio = True

    # メッセージの仕分けを行う、質問かどうか判別する
    event.classification = await ossans_navi_service.classify(thread_messages)
    logger.info(f"classify intent          {event.user_intent}")
    logger.info(f"classify intent_type     {event.user_intentions_type}")
    logger.info(f"classify who_to_talk_to  {event.who_to_talk_to}")
    logger.info(f"classify emotions        {event.user_emotions}")
    logger.info(f"classify knowledge_types {event.required_knowledge_types}")
    logger.info(f"classify slack_emojis    {event.slack_emoji_names}")

    if not event.is_need_response() and not event.is_mention:
        # 質問・相談ではなく、メンションされていない場合はここで終了
        if event.is_reply_to_ossans_navi():
            # OssansNavi へ話かけているならリアクションで返す
            await slack_service.add_reaction(event.channel_id, event.ts, event.slack_emoji_names)
            logger.info("Finished with reaction.")
        return

    # 定期的にイベントがキャンセルされていないか確認して、キャンセルされていれば終了する
    yield

    # メッセージ内に含まれるファイルをロードする
    # ## このタイミングまでファイルをロードしない理由
    # ossans_navi_service.classify 以前のフェーズは大量のメッセージが流入することを考慮が必要
    # 重い処理や費用の増加する処理は classify を通過して応答フェーズに入った後に実施する
    for (i, message) in enumerate(thread_messages):
        is_latest_message = i == len(thread_messages) - 1
        for file in message.files:
            if file.is_text or file.is_canvas or file.is_image:
                # テキストファイル、キャンバス、画像ファイルはロードする
                await ossans_navi_service.load_slack_file(file, user_client=False, load_file=True, load_vtt=False)
            elif file.is_video or file.is_audio:
                if is_latest_message and event.is_mention and config.LOAD_VIDEO_AUDIO_FILES:
                    # 動画や音声ファイルは負荷とコストが高いので以下の条件に適合する場合のみロードする
                    #   - 最新メッセージに添付されている
                    #   - OssansNavi がメンションされている
                    #   - 動画や音声ファイルのロードが有効になっている
                    await ossans_navi_service.load_slack_file(file, user_client=False, load_file=True, load_vtt=True)

    # 添付画像がある場合は画像の説明を取得する
    await ossans_navi_service.analyze_image_description(thread_messages)

    if event.is_need_additional_information:
        if ossans_navi_service.has_progress_reaction():
            # 処理中リアクションが付いている場合はリアクションを更新する
            await ossans_navi_service.do_progress_reaction(config.PROGRESS_REACTION_SEARCH)
        # Slack ワークスペースを検索するワードを生成してもらう
        # get_slack_searches() は Generator で処理単位ごとに yield している
        # なぜならば、呼び出し側で EVENT_GUARD.is_canceled() をチェックするタイミングを用意するためで、ループごとに確認してキャンセルされていれば終了する
        async for _ in ossans_navi_service.do_slack_searches(thread_messages=thread_messages):
            # 定期的にイベントがキャンセルされていないか確認して、キャンセルされていれば終了する
            yield

        if ossans_navi_service.has_progress_reaction():
            # 処理中リアクションが付いている場合はリアクションを更新する
            await ossans_navi_service.do_progress_reaction(config.PROGRESS_REACTION_REFINE)
        # slack_searches の結果から有用な情報を抽出するフェーズ（refine_slack_searches）
        # トークン数の上限があるので複数回に分けて実行して、大量の検索結果の中から必要な情報を絞り込む
        # RAG で入力する情報以外のトークン数を求めておく（システムプロンプトなど）、RAG で入力可能な情報を計算する為に使う
        async for _ in ossans_navi_service.refine_slack_searches(thread_messages=thread_messages):
            # 定期的にイベントがキャンセルされていないか確認して、キャンセルされていれば終了する
            yield

    if ossans_navi_service.has_progress_reaction():
        # 処理中リアクションが付いている場合はリアクションを更新する
        await ossans_navi_service.do_progress_reaction(config.PROGRESS_REACTION_LASTSHOT)

    # 集まった情報を元に返答を生成するフェーズ（lastshot）
    # GPT-4o で最終的な答えを生成する（GPT-4o mini で精査した情報を利用）
    lastshot_responses = await ossans_navi_service.lastshot(thread_messages=thread_messages)

    logger.info(f"{lastshot_responses=}")
    if len(lastshot_responses) == 0:
        # 返答が空のケースや期待するJSON形式じゃないパターン、回答の必要がないと判断
        logger.error("No valid response, Finished.")
        return

    lastshot_response = sorted(
        lastshot_responses,
        key=lambda v: len(v.text),
        reverse=True
    )[0]

    do_response = False
    if event.is_mention:
        # メンションされている場合は応答する
        do_response = True
    elif event.is_need_response():
        # 応答が必要とされるメッセージには、以下の条件に合致する場合のみ応答する
        # まず応答品質の判定をして、その結果に応じて応答するか決定する
        quality_check_response = await ossans_navi_service.quality_check(thread_messages, lastshot_response.text)
        if quality_check_response.user_intent is not None and quality_check_response.response_quality:
            # ユーザーに意図があり、かつ応答クオリティが高いと判断している場合は応答する
            do_response = True
        elif quality_check_response.user_intent is not None and event.is_reply_to_ossans_navi():
            # ユーザーに意図があり、かつ OssansNavi のメッセージの直後のメッセージの場合は応答する
            do_response = True

    # 定期的にイベントがキャンセルされていないか確認して、キャンセルされていれば終了する
    yield

    if do_response:
        # 処理中リアクションが付いている場合は削除する
        await ossans_navi_service.remove_progress_reaction()
        async with EVENT_GUARD:
            # 同タイミングでキャンセルされた場合に応答しないため、ロックした状態で応答処理をする
            if EVENT_GUARD.is_canceled(event):
                logger.info(f"Event canceled: {event.id()}({event.id_source})")
                logger.info("Finished.")
                return
            await slack_service.chat_post_message(
                channel=event.channel_id,
                thread_ts=event.thread_ts,
                text=(
                    await slack_service.disable_mention_if_not_active(
                        SlackService.convert_markdown_to_mrkdwn(lastshot_response.text)
                    )
                ),
                images=lastshot_response.images,
            )
            if config.RESPONSE_LOGGING_CHANNEL:
                await slack_service.chat_post_message(
                    config.RESPONSE_LOGGING_CHANNEL,
                    json.dumps({
                        "cost": ossans_navi_service.ai_service.models_usage.get_total_cost(),
                        "channel": event.channel_id,
                        "thread_ts": event.thread_ts,
                    }, ensure_ascii=False)
                )
            # 応答した場合はロックしたまま finish する
            # さもないと、応答しているのに同タイミングで is_queueed_or_running の判定が True となる可能性があるため
            EVENT_GUARD.finish(event)

    # ここまで至ると正常終了
    logger.info("Finished normally.")


async def main():
    args = sys.argv[1:]
    if "--production" in args:
        config.DEVELOPMENT_MODE = False
    if "--silent" in args:
        config.SILENT_MODE = True
    if "--unsafe" in args:
        config.SAFE_MODE = False

    logger.info(
        f"Strat in {"development" if config.DEVELOPMENT_MODE else "production"},"
        + f" {"silent" if config.SILENT_MODE else "no-silent"}, {"safe" if config.SAFE_MODE else "unsafe"} mode"
    )

    # config のダンプ
    logger.info("config dump:")
    for name in dir(config):
        if not re.search(r"^[A-Z]", name):
            continue
        if (value := getattr(config, name)) is None:
            continue
        if isinstance(value, str) and ("API_KEY" in name or "TOKEN" in name):
            value = "*" * len(str(value))
        logger.info(f"  {name}={value}")

    # 非同期実行が必要な初期化処理を実行
    await asyncio.gather(
        AiService.start(),
        slack_service.start(),
    )

    # TERM/INTシグナルにトラップして (graceful) shutdown を実行、以下のフローで終了する
    #   1. Slack サーバから WebSocket を切断して新規メッセージの受信を停止
    #   2. await event.wait() で停止した非同期処理を event.set() で終了させる
    #   3. graceful の場合は現在実行中のタスクが完了するまで最大 600秒待機する
    #   4. 終了する
    event = asyncio.Event()
    graceful = True

    def shutdown_handler(graceful_param: bool):
        def handler(signal, frame):
            nonlocal graceful
            graceful = graceful_param
            logger.info("Stopping Slack app...")
            event.set()
        return handler

    signal.signal(signal.SIGTERM, shutdown_handler(True))
    signal.signal(signal.SIGINT, shutdown_handler(False))
    await event.wait()

    # アプリの終了
    await slack_service.stop()
    logger.info("Slack app stopped.")
    if graceful:
        current_task = asyncio.current_task()
        pending = [t for t in asyncio.all_tasks() if t is not current_task]
        if pending:
            logger.info("Wait for current task to complete")
            try:
                await asyncio.wait_for(asyncio.gather(*pending), timeout=600)
                logger.info("All current tasks completed")
            except asyncio.TimeoutError:
                logger.info("Timeout waiting for current tasks to complete")
                EVENT_GUARD.terminate()
                logger.info("Events Terminated.")


# アプリを起動します
if __name__ == "__main__":
    asyncio.run(main())
