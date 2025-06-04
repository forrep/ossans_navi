import itertools
import json
import logging
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Final, Generator

from ossans_navi import config
from ossans_navi.service.ai_service import AiModels, AiService
from ossans_navi.service.ossans_navi_service import OssansNaviService
from ossans_navi.service.slack_service import EventGuard, SlackService
from ossans_navi.type.slack_type import SlackMessageEvent

_EXECUTOR: ThreadPoolExecutor
_EXECUTOR_WORKERS: Final[int] = 2
EVENT_GUARD = EventGuard()

slack_service = SlackService()
ai_service = AiService()

logger = logging.getLogger(__name__)


@slack_service.app.event("message")
def handle_message_events(say, event: dict[str, dict]):
    logger.info("event=" + json.dumps(event, ensure_ascii=False))
    message_event = SlackMessageEvent(event)

    def future_callback(v):
        logger.info(f"Event finished: {message_event.id()} ({message_event.channel_id},{message_event.thread_ts},{message_event.ts})")
        # 応答まで至ったパターンはロックを取ってから finish が必要なのですでに終了済み、重複して終了しても問題なし
        # それ以外パターンはまだ finish してないのでここで finish が必要
        EVENT_GUARD.finish(message_event)

    # ロックを取得してから処理キューに入れる処理
    # ロックしないと応答が二重になるケースが発生する
    with EVENT_GUARD:
        if message_event.is_message_post():
            # メッセージの投稿イベントの場合
            EVENT_GUARD.queue(message_event)
            logger.info(f"Event queued(message_post): {message_event.id()} ({message_event.channel_id},{message_event.thread_ts},{message_event.ts})")
        elif message_event.is_message_changed():
            # 更新イベントの場合
            if EVENT_GUARD.is_queueed_or_running(message_event):
                # 更新イベントは、元メッセージが QUEUED か RUNNING の場合（つまりまだ応答してない）場合に限ってキューに追加する
                # このパターンのためにロックが必要、ロックしてないと is_queueed_or_running 判定後に応答が終了する可能性もある
                EVENT_GUARD.queue(message_event)
                logger.info(
                    f"Event queued(message_changed): {message_event.id()} ({message_event.channel_id},{message_event.thread_ts},{message_event.ts})"
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
        future = _EXECUTOR.submit(do_ossans_navi_response_safe, say, message_event)
        future.add_done_callback(future_callback)


def do_ossans_navi_response_safe(say, event: SlackMessageEvent):
    with EVENT_GUARD:
        if EVENT_GUARD.is_canceled(event):
            logger.info(f"Event canceled: {event.id()} ({event.channel_id},{event.thread_ts},{event.ts})")
            logger.info("Finished.")
            return
        logger.info(f"Event started: {event.id()} ({event.channel_id},{event.thread_ts},{event.ts})")
        EVENT_GUARD.start(event)

    models = AiModels.new()
    try:
        # event の構築作業
        event.canceled_events.extend(EVENT_GUARD.get_canceled_events(event))
        if event.is_user:
            event.user = slack_service.get_user(event.user_id)
        else:
            event.user = slack_service.get_bot(event.bot_id)
        event.channel = slack_service.get_channel(event.channel_id)
        mentions = [slack_service.get_user(user_id) for user_id in event.mentions]
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

        for _ in do_ossans_navi_response(say, event, models):
            # yield のタイミングで処理が戻されるごとにイベントがキャンセルされていないか確認して、キャンセルされていれば終了する
            if EVENT_GUARD.is_canceled(event):
                logger.info(f"Event canceled: {event.id()}({event.channel_id},{event.thread_ts},{event.ts})")
                logger.info("Finished.")
                break
    except Exception as e:
        logger.error("do_ossans_navi_response return error")
        logger.error(e, exc_info=True)
        if event.is_mention:
            try:
                say(text=f"申し訳ありません、次のエラーが発生したため回答できません\n\n```{str(e)}```", thread_ts=event.thread_ts)
            except Exception as e:
                logger.error(e, exc_info=True)
    finally:
        logger.info(f"Usage Report (Total Cost: {models.get_total_cost():.4f})")
        for (model_type, model) in [("LOW_COST", models.low_cost), ("HIGH_QUALITY", models.high_quality)]:
            logger.info(f"  {model_type}: {model.name} (Cost: {model.get_total_cost():.4f})")
            logger.info(f"    tokens_in  = {model.tokens_in}")
            logger.info(f"    tokens_out = {model.tokens_out}")


def do_ossans_navi_response(say, event: SlackMessageEvent, models: AiModels) -> Generator[None, None, None]:
    ossans_navi_service = OssansNaviService(ai_service=ai_service, slack_service=slack_service, models=models, event=event)

    # yield で呼び出し元に戻すとイベントのキャンセルチェックをして、キャンセルされていれば終了する
    yield

    if event.is_dm() and ossans_navi_service.special_command():
        # DM の場合は special_command() を実行、そして True が返ってきたら special_command を実行しているので通常のメッセージ処理は終了する
        return

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
    if event.is_user:
        event.settings = slack_service.get_dm_info_with_ossans_navi(event.user.user_id)

    # スレッドでやりとりされた履歴メッセージを取得
    thread_messages = ossans_navi_service.get_thread_messages()
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

    # メッセージの仕分けを行う、質問かどうか判別する
    event.classification = ossans_navi_service.classify(thread_messages)
    logger.info(
        f"{event.user_intent=}, {event.user_intentions_type=}, {event.who_to_talk_to=}, "
        + f"{event.user_emotions=}, {event.required_knowledge_types=}, {event.slack_emoji_names=}"
    )

    if not event.is_need_response() and not event.is_mention:
        # 質問・相談ではなく、メンションされていない場合はここで終了
        if event.is_reply_to_ossans_navi():
            # OssansNavi へ話かけているならリアクションで返す
            slack_service.add_reaction(
                event.channel_id,
                event.ts,
                event.slack_emoji_names
            )
            logger.info("Finished with reaction.")
        return

    # 定期的にイベントがキャンセルされていないか確認して、キャンセルされていれば終了する
    yield

    # メッセージ内に含まれるファイルをロードする
    # ## このタイミングまでファイルをロードしない理由
    # ossans_navi_service.classify 以前のフェーズは大量のメッセージが流入することを考慮が必要
    # 重い処理や費用の増加する処理は classify を通過して応答フェーズに入った後に実施する
    for file in itertools.chain.from_iterable([message.files for message in thread_messages]):
        ossans_navi_service.load_slack_file(file, False)

    # 添付画像がある場合は画像の説明を取得する
    ossans_navi_service.analyze_image_description(thread_messages)

    if event.is_need_additional_information:
        # Slack ワークスペースを検索するワードを生成してもらう
        # get_slack_searches() は Generator で処理単位ごとに yield している
        # なぜならば、呼び出し側で EVENT_GUARD.is_canceled() をチェックするタイミングを用意するためで、ループごとに確認してキャンセルされていれば終了する
        for _ in ossans_navi_service.do_slack_searches(thread_messages=thread_messages):
            # 定期的にイベントがキャンセルされていないか確認して、キャンセルされていれば終了する
            yield

        # slack_searches の結果から有用な情報を抽出するフェーズ（refine_slack_searches）
        # トークン数の上限があるので複数回に分けて実行して、大量の検索結果の中から必要な情報を絞り込む
        # RAG で入力する情報以外のトークン数を求めておく（システムプロンプトなど）、RAG で入力可能な情報を計算する為に使う
        for _ in ossans_navi_service.refine_slack_searches(thread_messages=thread_messages):
            # 定期的にイベントがキャンセルされていないか確認して、キャンセルされていれば終了する
            yield

    # 集まった情報を元に返答を生成するフェーズ（lastshot）
    # GPT-4o で最終的な答えを生成する（GPT-4o mini で精査した情報を利用）
    lastshot_responses = ossans_navi_service.lastshot(thread_messages=thread_messages)

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
        quality_check_response = ossans_navi_service.quality_check(thread_messages, lastshot_response.text)
        if quality_check_response.user_intent is not None and quality_check_response.response_quality:
            # ユーザーに意図があり、かつ応答クオリティが高いと判断している場合は応答する
            do_response = True
        elif quality_check_response.user_intent is not None and event.is_reply_to_ossans_navi():
            # ユーザーに意図があり、かつ OssansNavi のメッセージの直後のメッセージの場合は応答する
            do_response = True

    # 定期的にイベントがキャンセルされていないか確認して、キャンセルされていれば終了する
    yield

    if do_response:
        with EVENT_GUARD:
            # 同タイミングでキャンセルされた場合に応答しないため、ロックした状態で応答処理をする
            if EVENT_GUARD.is_canceled(event):
                logger.info(f"Event canceled: {event.id()}({event.channel_id},{event.thread_ts},{event.ts})")
                logger.info("Finished.")
                return
            slack_service.chat_post_message(
                channel=event.channel_id,
                thread_ts=event.thread_ts,
                text=(
                    slack_service.disable_mention_if_not_active(
                        SlackService.convert_markdown_to_mrkdwn(lastshot_response.text)
                    )
                ),
                images=lastshot_response.images,
            )
            if config.RESPONSE_LOGGING_CHANNEL:
                slack_service.chat_post_message(
                    config.RESPONSE_LOGGING_CHANNEL,
                    json.dumps({
                        "cost": models.get_total_cost(),
                        "channel": event.channel_id,
                        "thread_ts": event.thread_ts,
                    }, ensure_ascii=False)
                )
            # 応答した場合はロックしたまま finish する
            # さもないと、応答しているのに同タイミングで is_queueed_or_running の判定が True となる可能性があるため
            EVENT_GUARD.finish(event)

    # ここまで至ると正常終了
    logger.info("Finished normally.")


# アプリを起動します
if __name__ == "__main__":
    args = sys.argv[1:]
    if "--production" in args:
        config.DEVELOPMENT_MODE = False
    if "--silent" in args:
        config.SILENT_MODE = True
    if "--unsafe" in args:
        config.SAFE_MODE = False

    with ThreadPoolExecutor(max_workers=_EXECUTOR_WORKERS, thread_name_prefix="Worker") as executor:
        _EXECUTOR = executor
        logger.info(
            f"Strat in {"development" if config.DEVELOPMENT_MODE else "production"},"
            + f" {"silent" if config.SILENT_MODE else "no-silent"}, {"safe" if config.SAFE_MODE else "unsafe"} mode"
        )
        slack_service.start()

        # TERMシグナルにトラップして graceful shutdown を実行、以下のフローで終了する
        #   1. Slack サーバから WebSocket を切断して新規メッセージの受信を停止
        #   2. event.wait() で停止したスレッドを event.set() で起こす（メインスレッドの処理は全て終了）
        #   3. ThreadPoolExecutor が with 句を抜けるタイミングで溜まったキューが全て実行される
        #   4. 終了する
        event = Event()

        def shutdown_handler(graceful: bool):
            def handler(signal, frame):
                print("Stopping Slack app...")
                slack_service.stop()
                print("Slack app stopped.")
                if not graceful:
                    EVENT_GUARD.terminate()
                    print("Events Terminated.")
                event.set()
            return handler

        signal.signal(signal.SIGTERM, shutdown_handler(True))
        signal.signal(signal.SIGINT, shutdown_handler(False))
        event.wait()
