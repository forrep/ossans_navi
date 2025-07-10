import datetime

from ossans_navi.type.slack_type import (SlackAuthTestAppResponse, SlackAuthTestBotResponse, SlackAuthTestUserResponse,
                                         SlackConversationsRepliesResponse, SlackMessageEvent, SlackReactionType, SlackSearchTerm)

from . import slack_messages_sample


def test_slack_search_term():
    t1 = SlackSearchTerm(frozenset(("ワード1", "ワード2", "ワード3",)), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), datetime.datetime(2024, 12, 8, 0, 0, 0, 0))
    t1_sub1 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), None, None)
    t1_sub2 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), None)
    t1_sub3 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), None, datetime.datetime(2024, 12, 8, 0, 0, 0, 0))
    t1_sub4 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), datetime.datetime(2024, 12, 8, 0, 0, 0, 0))
    t1_sub5 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), datetime.datetime(2024, 11, 7, 0, 0, 0, 0), datetime.datetime(2024, 12, 9, 0, 0, 0, 0))
    assert t1_sub1.is_subset(t1)
    assert t1_sub2.is_subset(t1)
    assert t1_sub3.is_subset(t1)
    assert t1_sub4.is_subset(t1)
    assert t1_sub5.is_subset(t1)

    t1_not_sub1 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), datetime.datetime(2024, 11, 9, 0, 0, 0, 0), None)
    t1_not_sub2 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), None, datetime.datetime(2024, 12, 7, 0, 0, 0, 0))
    assert not t1_not_sub1.is_subset(t1)
    assert not t1_not_sub2.is_subset(t1)

    o1_1 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), None, None)
    o1_2 = SlackSearchTerm(frozenset(("ワード1", "ワード2", "ワード3",)), None, None)
    assert sorted([o1_1, o1_2]) == sorted([o1_2, o1_1]) == [o1_1, o1_2]

    o2_1 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), None, None)
    o2_2 = SlackSearchTerm(frozenset(("ワード1", "ワーード2",)), None, None)
    assert sorted([o2_1, o2_2]) == sorted([o2_2, o2_1]) == [o2_1, o2_2]

    o3_1 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), None, None)
    o3_2 = SlackSearchTerm(frozenset(("ワード1", "ワーー2",)), None, None)
    assert sorted([o3_1, o3_2]) == sorted([o3_2, o3_1]) == [o3_1, o3_2]

    o4_1 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), None, None)
    o4_2 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), datetime.datetime(2024, 11, 9, 0, 0, 0, 0), None)
    o4_3 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), datetime.datetime(2024, 11, 9, 0, 0, 0, 0), datetime.datetime(2024, 12, 7, 0, 0, 0, 0))
    assert sorted([o4_1, o4_2, o4_3]) == sorted([o4_3, o4_2, o4_1]) == [o4_1, o4_2, o4_3]

    o5_1 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), datetime.datetime(2024, 11, 7, 0, 0, 0, 0), datetime.datetime(2024, 12, 5, 0, 0, 0, 0))
    o5_2 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), datetime.datetime(2024, 12, 6, 0, 0, 0, 0))
    o5_3 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), datetime.datetime(2024, 11, 9, 0, 0, 0, 0), datetime.datetime(2024, 12, 7, 0, 0, 0, 0))
    assert sorted([o5_1, o5_2, o5_3]) == sorted([o5_3, o5_2, o5_1]) == [o5_1, o5_2, o5_3]

    # 不正フォーマットが指定されたら None を返却する
    o6_1 = SlackSearchTerm.parse("ワード1 from:<@テストさんのユーザーID> ワード2")
    assert o6_1 is None

    o7_1 = SlackSearchTerm.parse("ワード1 ワード2")
    o7_2 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), None, None)
    assert o7_1 == o7_2

    o8_1 = SlackSearchTerm.parse("ワード1 ワード2 after:2024-11-09 before:2024-12-07")
    o8_2 = SlackSearchTerm(frozenset(("ワード1", "ワード2",)), datetime.datetime(2024, 11, 9, 0, 0, 0, 0), datetime.datetime(2024, 12, 7, 0, 0, 0, 0))
    assert o8_1 == o8_2


def test_slack_message_event():
    # メッセージイベント（通常）
    event_message = SlackMessageEvent(slack_messages_sample.message)
    assert event_message.channel_id == "C0000XXX0XX"
    assert event_message.text == "<@U0761XXXX6X> テレビCMをしていたのはいつですか？"
    assert event_message.user_id == "U0XXX00X0"
    assert event_message.ts == "1736207085.539649"
    assert event_message.event_ts == "1736207085.539649"
    assert event_message.thread_ts == "1736207085.539649"
    assert event_message.mentions == ["U0761XXXX6X"]
    assert not event_message.is_broadcast()
    assert not event_message.is_thread()
    assert event_message.is_message_post()
    assert not event_message.is_message_changed()
    assert not event_message.is_message_deleted()
    assert not event_message.is_open_channel()
    assert not event_message.is_dm()
    assert not event_message.is_mention_to_subteam()

    # 更新イベント（通常）
    event_message_changed = SlackMessageEvent(slack_messages_sample.message_changed)
    assert event_message_changed.channel_id == "C0000XXX0XX"
    assert event_message_changed.text == "<@U0761XXXX6X> テレビ・ラジオでCMをしていたのはいつでしたっけ？"
    assert event_message_changed.user_id == "U0XXX00X0"
    assert event_message_changed.ts == "1736207085.539649"
    assert event_message_changed.event_ts == "1736207110.000400"
    assert event_message_changed.thread_ts == "1736207085.539649"
    assert event_message_changed.mentions == ["U0761XXXX6X"]
    assert not event_message_changed.is_broadcast()
    assert not event_message_changed.is_thread()
    assert not event_message_changed.is_message_post()
    assert event_message_changed.is_message_changed()
    assert not event_message_changed.is_message_deleted()
    assert not event_message_changed.is_open_channel()
    assert not event_message_changed.is_dm()
    assert not event_message_changed.is_mention_to_subteam()

    # メッセージイベント（スレッド内）
    event_message_in_thread = SlackMessageEvent(slack_messages_sample.message_in_thread)
    assert event_message_in_thread.channel_id == "C0000XXX0XX"
    assert event_message_in_thread.text == "ラジオCMは"
    assert event_message_in_thread.user_id == "U0XXX00X0"
    assert event_message_in_thread.ts == "1736481285.683829"
    assert event_message_in_thread.event_ts == "1736481285.683829"
    assert event_message_in_thread.thread_ts == "1736207085.539649"
    assert event_message_in_thread.mentions == []
    assert not event_message_in_thread.is_broadcast()
    assert event_message_in_thread.is_thread()
    assert event_message_in_thread.is_message_post()
    assert not event_message_in_thread.is_message_changed()
    assert not event_message_in_thread.is_message_deleted()
    assert not event_message_in_thread.is_open_channel()
    assert not event_message_in_thread.is_dm()
    assert not event_message_in_thread.is_mention_to_subteam()

    # 更新イベント（スレッド内）
    event_message_in_thread_changed = SlackMessageEvent(slack_messages_sample.message_in_thread_changed)
    assert event_message_in_thread_changed.channel_id == "C0000XXX0XX"
    assert event_message_in_thread_changed.text == "ラジオCMはやってましたか？"
    assert event_message_in_thread_changed.user_id == "U0XXX00X0"
    assert event_message_in_thread_changed.ts == "1736481285.683829"
    assert event_message_in_thread_changed.event_ts == "1736481301.000200"
    assert event_message_in_thread_changed.thread_ts == "1736207085.539649"
    assert event_message_in_thread_changed.mentions == []
    assert not event_message_in_thread_changed.is_broadcast()
    assert event_message_in_thread_changed.is_thread()
    assert not event_message_in_thread_changed.is_message_post()
    assert event_message_in_thread_changed.is_message_changed()
    assert not event_message_in_thread_changed.is_message_deleted()
    assert not event_message_in_thread_changed.is_open_channel()
    assert not event_message_in_thread_changed.is_dm()
    assert not event_message_in_thread_changed.is_mention_to_subteam()

    # 削除イベント（スレッドのルートメッセージ）
    # 「This message was deleted.」というメッセージに置き換わる、メッセージ自体は消えない
    event_thread_root_message_deleted = SlackMessageEvent(slack_messages_sample.thread_root_message_deleted)
    assert event_thread_root_message_deleted.channel_id == "C0000XXX0XX"
    assert event_thread_root_message_deleted.text == "This message was deleted."
    assert event_thread_root_message_deleted.user_id == "U0XXX00X0"
    assert event_thread_root_message_deleted.ts == "1736207085.539649"
    assert event_thread_root_message_deleted.event_ts == "1738195030.000300"
    assert event_thread_root_message_deleted.thread_ts == "1736207085.539649"
    assert event_thread_root_message_deleted.mentions == []
    assert not event_thread_root_message_deleted.is_broadcast()
    assert event_thread_root_message_deleted.is_thread()
    assert not event_thread_root_message_deleted.is_message_post()
    assert not event_thread_root_message_deleted.is_message_changed()
    assert event_thread_root_message_deleted.is_message_deleted()
    assert not event_thread_root_message_deleted.is_open_channel()
    assert not event_thread_root_message_deleted.is_dm()
    assert not event_thread_root_message_deleted.is_mention_to_subteam()

    # 削除イベント（スレッドの子メッセージ）
    event_message_in_thread_deleted = SlackMessageEvent(slack_messages_sample.message_in_thread_deleted)
    assert event_message_in_thread_deleted.channel_id == "C0000XXX0XX"
    assert event_message_in_thread_deleted.text == "This message was deleted."
    assert event_message_in_thread_deleted.user_id == "U0XXX00X0"
    assert event_message_in_thread_deleted.ts == "1736481285.683829"
    assert event_message_in_thread_deleted.event_ts == "1738194975.000100"
    assert event_message_in_thread_deleted.thread_ts == "1736207085.539649"
    assert event_message_in_thread_deleted.mentions == []
    assert not event_message_in_thread_deleted.is_broadcast()
    assert event_message_in_thread_deleted.is_thread()
    assert not event_message_in_thread_deleted.is_message_post()
    assert not event_message_in_thread_deleted.is_message_changed()
    assert event_message_in_thread_deleted.is_message_deleted()
    assert not event_message_in_thread_deleted.is_open_channel()
    assert not event_message_in_thread_deleted.is_dm()
    assert not event_message_in_thread_deleted.is_mention_to_subteam()

    # メッセージイベント2（通常）
    event_message2 = SlackMessageEvent(slack_messages_sample.message2)
    assert event_message2.channel_id == "C0000XXX0XX"
    assert event_message2.text == "テストメッセージ"
    assert event_message2.user_id == "U0XXX00X0"
    assert event_message2.ts == "1738209531.687019"
    assert event_message2.event_ts == "1738209531.687019"
    assert event_message2.thread_ts == "1738209531.687019"
    assert event_message2.mentions == []
    assert not event_message2.is_broadcast()
    assert not event_message2.is_thread()
    assert event_message2.is_message_post()
    assert not event_message2.is_message_changed()
    assert not event_message2.is_message_deleted()
    assert not event_message2.is_open_channel()
    assert not event_message2.is_dm()
    assert not event_message2.is_mention_to_subteam()

    # 削除イベント2（非スレッド）
    event_message2_deleted = SlackMessageEvent(slack_messages_sample.message2_deleted)
    assert event_message2_deleted.channel_id == "C0000XXX0XX"
    assert event_message2_deleted.text == "This message was deleted."
    assert event_message2_deleted.user_id == "U0XXX00X0"
    assert event_message2_deleted.ts == "1738209531.687019"
    assert event_message2_deleted.event_ts == "1738209537.000400"
    assert event_message2_deleted.thread_ts == "1738209531.687019"
    assert event_message2_deleted.mentions == []
    assert not event_message2_deleted.is_broadcast()
    assert not event_message2_deleted.is_thread()
    assert not event_message2_deleted.is_message_post()
    assert not event_message2_deleted.is_message_changed()
    assert event_message2_deleted.is_message_deleted()
    assert not event_message2_deleted.is_open_channel()
    assert not event_message2_deleted.is_dm()
    assert not event_message2_deleted.is_mention_to_subteam()

    # メッセージイベント（ファイル送信）
    event_message2 = SlackMessageEvent(slack_messages_sample.file_share_message)
    assert event_message2.channel_id == "C0000XXX0XX"
    assert event_message2.text == "<@U0761XXXX6X> 添付したファイルの内容をそのまま復唱してください。"
    assert event_message2.user_id == "U0XXX00X0"
    assert event_message2.ts == "1738799450.020939"
    assert event_message2.event_ts == "1738799450.020939"
    assert event_message2.thread_ts == "1738799450.020939"
    assert event_message2.mentions == ["U0761XXXX6X"]
    assert not event_message2.is_broadcast()
    assert not event_message2.is_thread()
    assert event_message2.is_message_post()
    assert not event_message2.is_message_changed()
    assert not event_message2.is_message_deleted()
    assert not event_message2.is_open_channel()
    assert not event_message2.is_dm()
    assert not event_message2.is_mention_to_subteam()

    # 更新イベント（ファイル送信）
    event_message_changed = SlackMessageEvent(slack_messages_sample.file_share_message_changed)
    assert event_message_changed.channel_id == "C0000XXX0XX"
    assert event_message_changed.text == "<@U0761XXXX6X> 添付したファイルの内容をそのまま2回復唱してください。"
    assert event_message_changed.user_id == "U0XXX00X0"
    assert event_message_changed.ts == "1738799450.020939"
    assert event_message_changed.event_ts == "1738799457.000400"
    assert event_message_changed.thread_ts == "1738799450.020939"
    assert event_message_changed.mentions == ["U0761XXXX6X"]
    assert not event_message_changed.is_broadcast()
    assert not event_message_changed.is_thread()
    assert not event_message_changed.is_message_post()
    assert event_message_changed.is_message_changed()
    assert not event_message_changed.is_message_deleted()
    assert not event_message_changed.is_open_channel()
    assert not event_message_changed.is_dm()
    assert not event_message_changed.is_mention_to_subteam()

    # 削除イベント（ファイル送信）
    event_message2_deleted = SlackMessageEvent(slack_messages_sample.file_share_root_message_deleted)
    assert event_message2_deleted.channel_id == "C0000XXX0XX"
    assert event_message2_deleted.text == "This message was deleted."
    assert event_message2_deleted.user_id == "U0XXX00X0"
    assert event_message2_deleted.ts == "1738799450.020939"
    assert event_message2_deleted.event_ts == "1738799648.000600"
    assert event_message2_deleted.thread_ts == "1738799450.020939"
    assert event_message2_deleted.mentions == []
    assert not event_message2_deleted.is_broadcast()
    assert event_message2_deleted.is_thread()
    assert not event_message2_deleted.is_message_post()
    assert not event_message2_deleted.is_message_changed()
    assert event_message2_deleted.is_message_deleted()
    assert not event_message2_deleted.is_open_channel()
    assert not event_message2_deleted.is_dm()
    assert not event_message2_deleted.is_mention_to_subteam()

    # メッセージイベント（ボット）
    event_bot_message = SlackMessageEvent(slack_messages_sample.bot_message)
    assert event_bot_message.channel_id == "C056X111111"
    assert event_bot_message.text == "ありがとうございます。"
    assert not event_bot_message.is_user
    assert event_bot_message.is_bot
    assert event_bot_message.bot_id == "B08E00XXXX0"
    assert event_bot_message.ts == "1740126435.330219"
    assert event_bot_message.event_ts == "1740126435.330219"
    assert event_bot_message.thread_ts == "1740126435.330219"
    assert event_bot_message.mentions == []
    assert not event_bot_message.is_broadcast()
    assert not event_bot_message.is_thread()
    assert event_bot_message.is_message_post()
    assert not event_bot_message.is_message_changed()
    assert not event_bot_message.is_message_deleted()
    assert event_bot_message.is_open_channel()
    assert not event_bot_message.is_dm()
    assert not event_bot_message.is_mention_to_subteam()

    # メッセージイベント（ファイル送信・ボット）
    event_bot_message_file_share = SlackMessageEvent(slack_messages_sample.bot_message_file_share)
    assert event_bot_message_file_share.channel_id == "C0000XXX0XX"
    assert event_bot_message_file_share.text == "<@U0761XXXX6X> こんにちは"
    assert not event_bot_message_file_share.is_user
    assert event_bot_message_file_share.is_bot
    assert event_bot_message_file_share.bot_id == "B08E00XXXX0"
    assert event_bot_message_file_share.ts == "1740455797.032319"
    assert event_bot_message_file_share.event_ts == "1740455797.032319"
    assert event_bot_message_file_share.thread_ts == "1740455797.032319"
    assert event_bot_message_file_share.mentions == ["U0761XXXX6X"]
    assert not event_bot_message_file_share.is_broadcast()
    assert not event_bot_message_file_share.is_thread()
    assert event_bot_message_file_share.is_message_post()
    assert not event_bot_message_file_share.is_message_changed()
    assert not event_bot_message_file_share.is_message_deleted()
    assert not event_bot_message_file_share.is_open_channel()
    assert not event_bot_message_file_share.is_dm()
    assert not event_bot_message_file_share.is_mention_to_subteam()


def test_slack_auth_test():
    response_app = SlackAuthTestAppResponse(
        **(
            {
                "ok": True,
                "app_name": "dev_ossans_navi",
                "app_id": "A075111XXXX"
            }
        )
    )
    assert response_app.ok
    assert response_app.app_name == "dev_ossans_navi"
    assert response_app.app_id == "A075111XXXX"

    response_user = SlackAuthTestUserResponse(
        **(
            {
                "ok": True,
                "url": "https://example.slack.com/",
                "team": "example",
                "user": "hayama",
                "team_id": "T02111XXX",
                "user_id": "U7C111XXX",
                "is_enterprise_install": False
            }
        )
    )
    assert response_user.ok
    assert response_user.url == "https://example.slack.com/"
    assert response_user.team == "example"
    assert response_user.user == "hayama"
    assert response_user.team_id == "T02111XXX"
    assert response_user.user_id == "U7C111XXX"

    response_bot = SlackAuthTestBotResponse(
        **(
            {
                "ok": True,
                "url": "https://example.slack.com/",
                "team": "example",
                "user": "ossansnavidev",
                "team_id": "T02111XXX",
                "user_id": "U0761111XXX",
                "bot_id": "B075T111XXX",
                "is_enterprise_install": False
            }
        )
    )
    assert response_bot.ok
    assert response_bot.url == "https://example.slack.com/"
    assert response_bot.team == "example"
    assert response_bot.user == "ossansnavidev"
    assert response_bot.team_id == "T02111XXX"
    assert response_bot.user_id == "U0761111XXX"
    assert response_bot.bot_id == "B075T111XXX"


def test_slack_conversations_replies():
    response1 = SlackConversationsRepliesResponse(
        **(
            {
                "ok": True,
                "messages": [
                    {
                        "user": "U7CAL37D0",
                        "type": "message",
                        "ts": "1717984816.912149",
                        "client_msg_id": "6aca2cf7-6744-4584-99e1-17b6e7875ef8",
                        "text": "<@U06NDV477T2> あなたの名前を教えてください",
                        "team": "T02L3BLC1",
                        "thread_ts": "1717984816.912149",
                        "reply_count": 1,
                        "reply_users_count": 1,
                        "latest_reply": "1717984901.384369",
                        "reply_users": [
                            "U0761EBMQ6S"
                        ],
                        "is_locked": False,
                        "subscribed": False,
                        "blocks": [
                            {
                                "type": "rich_text",
                                "block_id": "s0JXq",
                                "elements": [
                                    {
                                        "type": "rich_text_section",
                                        "elements": [
                                            {
                                                "type": "user",
                                                "user_id": "U06NDV477T2"
                                            },
                                            {
                                                "type": "text",
                                                "text": " あなたの名前を教えてください"
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "user": "U0761EBMQ6S",
                        "type": "message",
                        "ts": "1717984901.384369",
                        "bot_id": "B075TGHDXHD",
                        "app_id": "A075TGF8U4F",
                        "text": "私の名前は「おっさんずナビ」または「ossans_navi」です。よろしくお願いします！:blush:\n",
                        "team": "T02L3BLC1",
                        "bot_profile": {
                            "id": "B075TGHDXHD",
                            "deleted": False,
                            "name": "dev_ossans_navi",
                            "updated": 1718237139,
                            "app_id": "A075TGF8U4F",
                            "user_id": "U0761EBMQ6S",
                            "icons": {
                                "image_36": "https://avatars.slack-edge.com/2024-06-12/7261233488758_ba2838395cd61c1c0866_36.png",
                                "image_48": "https://avatars.slack-edge.com/2024-06-12/7261233488758_ba2838395cd61c1c0866_48.png",
                                "image_72": "https://avatars.slack-edge.com/2024-06-12/7261233488758_ba2838395cd61c1c0866_72.png"
                            },
                            "team_id": "T02L3BLC1"
                        },
                        "thread_ts": "1717984816.912149",
                        "parent_user_id": "U7CAL37D0",
                        "blocks": [
                            {
                                "type": "rich_text",
                                "block_id": "VcUwK",
                                "elements": [
                                    {
                                        "type": "rich_text_section",
                                        "elements": [
                                            {
                                                "type": "text",
                                                "text": "私の名前は「おっさんずナビ」または「ossans_navi」です。よろしくお願いします！"
                                            },
                                            {
                                                "type": "emoji",
                                                "name": "blush",
                                                "unicode": "1f60a"
                                            }
                                        ]
                                    }
                                ]
                            }
                        ],
                        "reactions": [
                            {
                                "name": "+1",
                                "users": [
                                    "U7CAL37D0"
                                ],
                                "count": 1
                            }
                        ]
                    }
                ],
                "has_more": False
            }
        )
    )
    assert response1.ok
    assert response1.messages[0].ts == "1717984816.912149"
    assert response1.messages[0].thread_ts == "1717984816.912149"
    assert response1.messages[0].user == "U7CAL37D0"
    assert response1.messages[0].bot_id is None
    assert response1.messages[0].text == "<@U06NDV477T2> あなたの名前を教えてください"
    assert len(response1.messages[0].attachments) == 0
    assert len(response1.messages[0].files) == 0
    assert len(response1.messages[0].reactions) == 0
    assert response1.messages[1].ts == "1717984901.384369"
    assert response1.messages[1].thread_ts == "1717984816.912149"
    assert response1.messages[1].user == "U0761EBMQ6S"
    assert response1.messages[1].bot_id == "B075TGHDXHD"
    assert response1.messages[1].text == "私の名前は「おっさんずナビ」または「ossans_navi」です。よろしくお願いします！:blush:\n"
    assert response1.messages[1].reactions == [SlackReactionType(name="+1", users=["U7CAL37D0"], count=1)]

    response2 = SlackConversationsRepliesResponse(
        **(
            {
                "ok": True,
                "messages": [
                    {
                        "user": "USLACKBOT",
                        "channel": "CXXXXXXX",
                        "room": {
                            "id": "RXXXXXXXX",
                            "name": "",
                            "media_server": "",
                            "created_by": "UXXXXXXX",
                            "date_start": 1745387799,
                            "date_end": 1745389769,
                            "participants": [],
                            "participant_history": [
                                "UXXXXXXX",
                                "UYYYYYYY",
                                "UZZZZZZZ",
                                "UWWWWWWW"
                            ],
                            "participants_events": {
                                "UXXXXXXX": {
                                    "user_team": {},
                                    "joined": True,
                                    "camera_on": False,
                                    "camera_off": False,
                                    "screenshare_on": False,
                                    "screenshare_off": False
                                },
                                "UYYYYYYY": {
                                    "user_team": {},
                                    "joined": True,
                                    "camera_on": False,
                                    "camera_off": False,
                                    "screenshare_on": False,
                                    "screenshare_off": False
                                },
                                "UZZZZZZZ": {
                                    "user_team": {},
                                    "joined": True,
                                    "camera_on": False,
                                    "camera_off": False,
                                    "screenshare_on": False,
                                    "screenshare_off": False
                                },
                                "UWWWWWWW": {
                                    "user_team": {},
                                    "joined": True,
                                    "camera_on": False,
                                    "camera_off": False,
                                    "screenshare_on": False,
                                    "screenshare_off": False
                                }
                            },
                            "participants_camera_on": [],
                            "participants_camera_off": [],
                            "participants_screenshare_on": [],
                            "participants_screenshare_off": [],
                            "canvas_thread_ts": "1745387799.160249",
                            "thread_root_ts": "1745387799.160249",
                            "channels": [
                                "CXXXXXXX"
                            ],
                            "is_dm_call": False,
                            "was_rejected": False,
                            "was_missed": False,
                            "was_accepted": False,
                            "has_ended": True,
                            "background_id": "ARIZONA",
                            "canvas_background": "ARIZONA",
                            "is_prewarmed": False,
                            "is_scheduled": False,
                            "recording": {
                                "can_record_summary": "unavailable"
                            },
                            "locale": "ja-JP",
                            "attached_file_ids": [],
                            "media_backend_type": "free_willy",
                            "display_id": "",
                            "external_unique_id": "e05fb6c4-6342-4ead-b07a-b3a518df2713",
                            "app_id": "A00",
                            "call_family": "huddle",
                            "pending_invitees": {},
                            "last_invite_status_by_user": {},
                            "knocks": {},
                            "huddle_link": "https://example.com/huddle/TXXXXXXX/CXXXXXXX"
                        },
                        "no_notifications": True,
                        "permalink": "https://example.com/call/RXXXXXXXX",
                        "subtype": "huddle_thread",
                        "type": "message",
                        "ts": "1745387799.160249",
                        "edited": {
                            "user": "USLACKBOT",
                            "ts": "1745389769.000000"
                        },
                        "text": "",
                        "team": "TXXXXXXX",
                        "thread_ts": "1745387799.160249",
                        "reply_count": 4,
                        "reply_users_count": 2,
                        "latest_reply": "1745389431.931479",
                        "reply_users": [
                            "UXXXXXXX",
                            "UYYYYYYY"
                        ],
                        "is_locked": False,
                        "subscribed": False,
                        "blocks": [
                            {
                                "type": "rich_text",
                                "block_id": "VZ2F3",
                                "elements": [
                                    {
                                        "type": "rich_text_section",
                                        "elements": [
                                            {
                                                "type": "text",
                                                "text": "ハドルミーティングが開始しました"
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "subtype": "thread_broadcast",
                        "user": "UXXXXXXX",
                        "thread_ts": "1745387799.160249",
                        "root": {
                            "user": "USLACKBOT",
                            "channel": "CXXXXXXX",
                            "room": {
                                "id": "RXXXXXXXX",
                                "name": "",
                                "media_server": "",
                                "created_by": "UXXXXXXX",
                                "date_start": 1745387799,
                                "date_end": 1745389769,
                                "participants": [],
                                "participant_history": [
                                    "UXXXXXXX",
                                    "UYYYYYYY",
                                    "UZZZZZZZ",
                                    "UWWWWWWW"
                                ],
                                "participants_events": {
                                    "UXXXXXXX": {
                                        "user_team": {},
                                        "joined": True,
                                        "camera_on": False,
                                        "camera_off": False,
                                        "screenshare_on": False,
                                        "screenshare_off": False
                                    },
                                    "UYYYYYYY": {
                                        "user_team": {},
                                        "joined": True,
                                        "camera_on": False,
                                        "camera_off": False,
                                        "screenshare_on": False,
                                        "screenshare_off": False
                                    },
                                    "UZZZZZZZ": {
                                        "user_team": {},
                                        "joined": True,
                                        "camera_on": False,
                                        "camera_off": False,
                                        "screenshare_on": False,
                                        "screenshare_off": False
                                    },
                                    "UWWWWWWW": {
                                        "user_team": {},
                                        "joined": True,
                                        "camera_on": False,
                                        "camera_off": False,
                                        "screenshare_on": False,
                                        "screenshare_off": False
                                    }
                                },
                                "participants_camera_on": [],
                                "participants_camera_off": [],
                                "participants_screenshare_on": [],
                                "participants_screenshare_off": [],
                                "canvas_thread_ts": "1745387799.160249",
                                "thread_root_ts": "1745387799.160249",
                                "channels": [
                                    "CXXXXXXX"
                                ],
                                "is_dm_call": False,
                                "was_rejected": False,
                                "was_missed": False,
                                "was_accepted": False,
                                "has_ended": True,
                                "background_id": "ARIZONA",
                                "canvas_background": "ARIZONA",
                                "is_prewarmed": False,
                                "is_scheduled": False,
                                "recording": {
                                    "can_record_summary": "unavailable"
                                },
                                "locale": "ja-JP",
                                "attached_file_ids": [],
                                "media_backend_type": "free_willy",
                                "display_id": "",
                                "external_unique_id": "e05fb6c4-6342-4ead-b07a-b3a518df2713",
                                "app_id": "A00",
                                "call_family": "huddle",
                                "pending_invitees": {},
                                "last_invite_status_by_user": {},
                                "knocks": {},
                                "huddle_link": "https://example.com/huddle/TXXXXXXX/CXXXXXXX"
                            },
                            "no_notifications": True,
                            "permalink": "https://example.com/call/RXXXXXXXX",
                            "subtype": "huddle_thread",
                            "type": "message",
                            "ts": "1745387799.160249",
                            "edited": {
                                "user": "USLACKBOT",
                                "ts": "1745389769.000000"
                            },
                            "text": "",
                            "team": "TXXXXXXX",
                            "thread_ts": "1745387799.160249",
                            "reply_count": 4,
                            "reply_users_count": 2,
                            "latest_reply": "1745389431.931479",
                            "reply_users": [
                                "UXXXXXXX",
                                "UYYYYYYY"
                            ],
                            "is_locked": False,
                            "subscribed": False,
                            "blocks": [
                                {
                                    "type": "rich_text",
                                    "block_id": "VZ2F3",
                                    "elements": [
                                        {
                                            "type": "rich_text_section",
                                            "elements": [
                                                {
                                                    "type": "text",
                                                    "text": "ハドルミーティングが開始しました"
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        },
                        "type": "message",
                        "ts": "1745389271.537969",
                        "client_msg_id": "304f2f73-f25c-471d-9956-2c4d6659a4bd",
                        "text": "<https://example.com/dp/ITEM1>",
                        "attachments": [
                            {
                                "from_url": "https://example.com/dp/ITEM1",
                                "service_icon": "https://example.com/favicon.ico",
                                "id": 1,
                                "original_url": "https://example.com/dp/ITEM1",
                                "fallback": "Anker Prime Power Bank ...",
                                "text": "Anker Prime Power Bank ...",
                                "service_name": "example.com"
                            }
                        ],
                        "blocks": [
                            {
                                "type": "rich_text",
                                "block_id": "+l+Wa",
                                "elements": [
                                    {
                                        "type": "rich_text_section",
                                        "elements": [
                                            {
                                                "type": "link",
                                                "url": "https://example.com/dp/ITEM1"
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "user": "UYYYYYYY",
                        "type": "message",
                        "ts": "1745389299.990499",
                        "client_msg_id": "f5fa3f54-7e88-46aa-bdf9-2cd26e652d14",
                        "text": "<https://example.com/dp/ITEM2>",
                        "team": "TXXXXXXX",
                        "thread_ts": "1745387799.160249",
                        "parent_user_id": "USLACKBOT",
                        "attachments": [
                            {
                                "from_url": "https://example.com/dp/ITEM2",
                                "service_icon": "https://example.com/favicon.ico",
                                "id": 1,
                                "original_url": "https://example.com/dp/ITEM2",
                                "fallback": "UGREEN Nexode モバイルバッテリー ...",
                                "text": "UGREEN Nexode モバイルバッテリー ...",
                                "service_name": "example.com"
                            }
                        ],
                        "blocks": [
                            {
                                "type": "rich_text",
                                "block_id": "kbIkx",
                                "elements": [
                                    {
                                        "type": "rich_text_section",
                                        "elements": [
                                            {
                                                "type": "link",
                                                "url": "https://example.com/dp/ITEM2"
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "user": "UYYYYYYY",
                        "type": "message",
                        "ts": "1745389392.318809",
                        "client_msg_id": "5e34a7bb-2afa-4206-bf95-58d2823a9f64",
                        "text": "<https://example.com/dp/ITEM3>",
                        "team": "TXXXXXXX",
                        "thread_ts": "1745387799.160249",
                        "parent_user_id": "USLACKBOT",
                        "attachments": [
                            {
                                "from_url": "https://example.com/dp/ITEM3",
                                "service_icon": "https://example.com/favicon.ico",
                                "id": 1,
                                "original_url": "https://example.com/dp/ITEM3",
                                "fallback": "CIO NovaPort DUOⅡ 65W ...",
                                "text": "CIO NovaPort DUOⅡ 65W ...",
                                "service_name": "example.com"
                            }
                        ],
                        "blocks": [
                            {
                                "type": "rich_text",
                                "block_id": "/XMBz",
                                "elements": [
                                    {
                                        "type": "rich_text_section",
                                        "elements": [
                                            {
                                                "type": "link",
                                                "url": "https://example.com/dp/ITEM3"
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "text": "",
                        "files": [
                            {
                                "id": "FXXXXXXX",
                                "created": 1745389429,
                                "timestamp": 1745389429,
                                "name": "image.png",
                                "title": "image.png",
                                "mimetype": "image/png",
                                "filetype": "png",
                                "pretty_type": "PNG",
                                "user": "UYYYYYYY",
                                "user_team": "TXXXXXXX",
                                "editable": False,
                                "size": 232277,
                                "mode": "hosted",
                                "is_external": False,
                                "external_type": "",
                                "is_public": True,
                                "public_url_shared": False,
                                "display_as_bot": False,
                                "username": "",
                                "url_private": "https://example.com/files-pri/TXXXXXXX-FXXXXXXX/image.png",
                                "url_private_download": "https://example.com/files-pri/TXXXXXXX-FXXXXXXX/download/image.png",
                                "media_display_type": "unknown",
                                "thumb_64": "https://example.com/files-tmb/TXXXXXXX-FXXXXXXX/image_64.png",
                                "thumb_80": "https://example.com/files-tmb/TXXXXXXX-FXXXXXXX/image_80.png",
                                "thumb_360": "https://example.com/files-tmb/TXXXXXXX-FXXXXXXX/image_360.png",
                                "thumb_360_w": 278,
                                "thumb_360_h": 360,
                                "thumb_480": "https://example.com/files-tmb/TXXXXXXX-FXXXXXXX/image_480.png",
                                "thumb_480_w": 371,
                                "thumb_480_h": 480,
                                "thumb_160": "https://example.com/files-tmb/TXXXXXXX-FXXXXXXX/image_160.png",
                                "original_w": 555,
                                "original_h": 718,
                                "thumb_tiny": "AwAwACWcDNGDSgUuBTAbRTsCigBtFOooABS9ao3bsyLkY57U6wYh8epFAFzFIQcE9h3qaXIQkelM3DyMn05pAR5ozUCSMFAwKXzW9BTAe0av95QaFhRCCq4PtUlIaABiWXBYkemaZsUKQBwaf2pOaAIfKj/uijyo/wC6KkINJzQB/9k=",
                                "permalink": "https://example.com/files/UXXXXXXX/FXXXXXXX/image.png",
                                "permalink_public": "https://example.com/TXXXXXXX-FXXXXXXX-abcdefg",
                                "is_starred": False,
                                "skipped_shares": True,
                                "has_rich_preview": False,
                                "file_access": "visible"
                            }
                        ],
                        "upload": False,
                        "user": "UYYYYYYY",
                        "display_as_bot": False,
                        "type": "message",
                        "ts": "1745389431.931479",
                        "client_msg_id": "7f17c724-1a33-46b0-a407-11b26e24d832",
                        "thread_ts": "1745387799.160249",
                        "parent_user_id": "USLACKBOT"
                    }
                ],
                "has_more": False
            }
        )
    )

    assert response2.ok
    assert len(response2.messages) == 5
    assert response2.messages[0].ts == "1745387799.160249"
    assert response2.messages[0].thread_ts == "1745387799.160249"
    assert response2.messages[0].user == "USLACKBOT"
    assert response2.messages[0].bot_id is None
    assert response2.messages[0].text == ""
    assert len(response2.messages[0].attachments) == 0
    assert len(response2.messages[0].files) == 0
    assert len(response2.messages[0].reactions) == 0
    assert len(response2.messages[1].attachments) == 1
    assert len(response2.messages[2].attachments) == 1
    assert len(response2.messages[3].attachments) == 1
    assert len(response2.messages[4].files) == 1
