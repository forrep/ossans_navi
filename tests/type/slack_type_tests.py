import datetime

from ossans_navi.type.slack_type import SlackMessageEvent, SlackSearchTerm


def test_slack_search_term():
    t1 = SlackSearchTerm(("ワード1", "ワード2", "ワード3",), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), datetime.datetime(2024, 12, 8, 0, 0, 0, 0))
    t1_sub1 = SlackSearchTerm(("ワード1", "ワード2",), None, None)
    t1_sub2 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), None)
    t1_sub3 = SlackSearchTerm(("ワード1", "ワード2",), None, datetime.datetime(2024, 12, 8, 0, 0, 0, 0))
    t1_sub4 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), datetime.datetime(2024, 12, 8, 0, 0, 0, 0))
    t1_sub5 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 7, 0, 0, 0, 0), datetime.datetime(2024, 12, 9, 0, 0, 0, 0))
    assert t1_sub1.is_subset(t1)
    assert t1_sub2.is_subset(t1)
    assert t1_sub3.is_subset(t1)
    assert t1_sub4.is_subset(t1)
    assert t1_sub5.is_subset(t1)

    t1_not_sub1 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 9, 0, 0, 0, 0), None)
    t1_not_sub2 = SlackSearchTerm(("ワード1", "ワード2",), None, datetime.datetime(2024, 12, 7, 0, 0, 0, 0))
    assert not t1_not_sub1.is_subset(t1)
    assert not t1_not_sub2.is_subset(t1)

    o1_1 = SlackSearchTerm(("ワード1", "ワード2",), None, None)
    o1_2 = SlackSearchTerm(("ワード1", "ワード2", "ワード3",), None, None)
    assert sorted([o1_1, o1_2]) == sorted([o1_2, o1_1]) == [o1_1, o1_2]

    o2_1 = SlackSearchTerm(("ワード1", "ワード2",), None, None)
    o2_2 = SlackSearchTerm(("ワード1", "ワーード2",), None, None)
    assert sorted([o2_1, o2_2]) == sorted([o2_2, o2_1]) == [o2_1, o2_2]

    o3_1 = SlackSearchTerm(("ワード1", "ワード2",), None, None)
    o3_2 = SlackSearchTerm(("ワード1", "ワーー2",), None, None)
    assert sorted([o3_1, o3_2]) == sorted([o3_2, o3_1]) == [o3_1, o3_2]

    o4_1 = SlackSearchTerm(("ワード1", "ワード2",), None, None)
    o4_2 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 9, 0, 0, 0, 0), None)
    o4_3 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 9, 0, 0, 0, 0), datetime.datetime(2024, 12, 7, 0, 0, 0, 0))
    assert sorted([o4_1, o4_2, o4_3]) == sorted([o4_3, o4_2, o4_1]) == [o4_1, o4_2, o4_3]

    o5_1 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 7, 0, 0, 0, 0), datetime.datetime(2024, 12, 5, 0, 0, 0, 0))
    o5_2 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), datetime.datetime(2024, 12, 6, 0, 0, 0, 0))
    o5_3 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 9, 0, 0, 0, 0), datetime.datetime(2024, 12, 7, 0, 0, 0, 0))
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
    # 通常メッセージ
    message = {
        "user": "U0XXX00X0",
        "type": "message",
        "ts": "1736207085.539649",
        "client_msg_id": "78fc431e-d911-4d96-b466-f00d74e0a5a9",
        "text": "<@U0761XXXX6X> テレビCMをしていたのはいつですか？",
        "team": "T02X3XXX1",
        "blocks": [
            {
                "type": "rich_text",
                "block_id": "/xkc2",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [
                            {
                                "type": "user",
                                "user_id": "U0761XXXX6X"
                            },
                            {
                                "type": "text",
                                "text": " テレビCMをしていたのはいつですか？"
                            }
                        ]
                    }
                ]
            }
        ],
        "channel": "C0000XXX0XX",
        "event_ts": "1736207085.539649",
        "channel_type": "group"
    }
    event_message = SlackMessageEvent(message)
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

    # 通常メッセージを更新
    message_changed = {
        "type": "message",
        "subtype": "message_changed",
        "message": {
            "user": "U0XXX00X0",
            "type": "message",
            "edited": {
                "user": "U0XXX00X0",
                "ts": "1736207110.000000"
            },
            "client_msg_id": "78fc431e-d911-4d96-b466-f00d74e0a5a9",
            "text": "<@U0761XXXX6X> テレビ・ラジオでCMをしていたのはいつでしたっけ？",
            "team": "T02X3XXX1",
            "blocks": [
                {
                    "type": "rich_text",
                    "block_id": "S4W7i",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "user",
                                    "user_id": "U0761XXXX6X"
                                },
                                {
                                    "type": "text",
                                    "text": " テレビ・ラジオでCMをしていたのはいつでしたっけ？"
                                }
                            ]
                        }
                    ]
                }
            ],
            "ts": "1736207085.539649",
            "source_team": "T02X3XXX1",
            "user_team": "T02X3XXX1"
        },
        "previous_message": {
            "user": "U0XXX00X0",
            "type": "message",
            "ts": "1736207085.539649",
            "client_msg_id": "78fc431e-d911-4d96-b466-f00d74e0a5a9",
            "text": "<@U0761XXXX6X> テレビCMをしていたのはいつですか？",
            "team": "T02X3XXX1",
            "blocks": [
                {
                    "type": "rich_text",
                    "block_id": "/xkc2",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "user",
                                    "user_id": "U0761XXXX6X"
                                },
                                {
                                    "type": "text",
                                    "text": " テレビCMをしていたのはいつですか？"
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        "channel": "C0000XXX0XX",
        "hidden": True,
        "ts": "1736207110.000400",
        "event_ts": "1736207110.000400",
        "channel_type": "group"
    }
    event_message_changed = SlackMessageEvent(message_changed)
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

    # スレッド内の通常メッセージ
    message_in_thread = {
        "user": "U0XXX00X0",
        "type": "message",
        "ts": "1736481285.683829",
        "client_msg_id": "51e3533b-6fd8-4705-ae34-045ee74db354",
        "text": "ラジオCMは",
        "team": "T02X3XXX1",
        "thread_ts": "1736207085.539649",
        "parent_user_id": "U0XXX00X0",
        "blocks": [
            {
                "type": "rich_text",
                "block_id": "Bl6l6",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [
                            {
                                "type": "text",
                                "text": "ラジオCMは"
                            }
                        ]
                    }
                ]
            }
        ],
        "channel": "C0000XXX0XX",
        "event_ts": "1736481285.683829",
        "channel_type": "group"
    }
    event_message_in_thread = SlackMessageEvent(message_in_thread)
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

    # スレッド内の通常メッセージを更新
    message_in_thread_changed = {
        "type": "message",
        "subtype": "message_changed",
        "message": {
            "user": "U0XXX00X0",
            "type": "message",
            "edited": {
                "user": "U0XXX00X0",
                "ts": "1736481301.000000"
            },
            "client_msg_id": "51e3533b-6fd8-4705-ae34-045ee74db354",
            "text": "ラジオCMはやってましたか？",
            "team": "T02X3XXX1",
            "thread_ts": "1736207085.539649",
            "parent_user_id": "U0XXX00X0",
            "blocks": [
                {
                    "type": "rich_text",
                    "block_id": "tES24",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "ラジオCMはやってましたか？"
                                }
                            ]
                        }
                    ]
                }
            ],
            "ts": "1736481285.683829",
            "source_team": "T02X3XXX1",
            "user_team": "T02X3XXX1"
        },
        "previous_message": {
            "user": "U0XXX00X0",
            "type": "message",
            "ts": "1736481285.683829",
            "client_msg_id": "51e3533b-6fd8-4705-ae34-045ee74db354",
            "text": "ラジオCMは",
            "team": "T02X3XXX1",
            "thread_ts": "1736207085.539649",
            "parent_user_id": "U0XXX00X0",
            "blocks": [
                {
                    "type": "rich_text",
                    "block_id": "Bl6l6",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "ラジオCMは"
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        "channel": "C0000XXX0XX",
        "hidden": True,
        "ts": "1736481301.000200",
        "event_ts": "1736481301.000200",
        "channel_type": "group"
    }
    event_message_in_thread_changed = SlackMessageEvent(message_in_thread_changed)
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

    # スレッドのルートメッセージが削除された場合
    # 「This message was deleted.」というメッセージに置き換わる、メッセージ自体は消えない
    thread_root_message_deleted = {
        "type": "message",
        "subtype": "message_changed",
        "message": {
            "subtype": "tombstone",
            "text": "This message was deleted.",
            "user": "USLACKBOT",
            "hidden": True,
            "type": "message",
            "thread_ts": "1736207085.539649",
            "reply_count": 2,
            "reply_users_count": 1,
            "latest_reply": "1736481392.096479",
            "reply_users": [
                "U0761XXXX6X"
            ],
            "is_locked": False,
            "ts": "1736207085.539649"
        },
        "previous_message": {
            "user": "U0XXX00X0",
            "type": "message",
            "ts": "1736207085.539649",
            "edited": {
                "user": "U0XXX00X0",
                "ts": "1736207110.000000"
            },
            "client_msg_id": "78fc431e-d911-4d96-b466-f00d74e0a5a9",
            "text": "<@U0761XXXX6X> テレビ・ラジオでCMをしていたのはいつでしたっけ？",
            "team": "T02X3XXX1",
            "thread_ts": "1736207085.539649",
            "reply_count": 2,
            "reply_users_count": 1,
            "latest_reply": "1736481392.096479",
            "reply_users": [
                "U0761XXXX6X"
            ],
            "is_locked": False,
            "subscribed": True,
            "last_read": "1736481392.096479",
            "blocks": [
                {
                    "type": "rich_text",
                    "block_id": "S4W7i",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "user",
                                    "user_id": "U0761XXXX6X"
                                },
                                {
                                    "type": "text",
                                    "text": " テレビ・ラジオでCMをしていたのはいつでしたっけ？"
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        "channel": "C0000XXX0XX",
        "hidden": True,
        "ts": "1738195030.000300",
        "event_ts": "1738195030.000300",
        "channel_type": "group"
    }
    event_thread_root_message_deleted = SlackMessageEvent(thread_root_message_deleted)
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

    message_in_thread_deleted = {
        "type": "message",
        "subtype": "message_deleted",
        "previous_message": {
            "user": "U0XXX00X0",
            "type": "message",
            "ts": "1736481285.683829",
            "edited": {
                "user": "U0XXX00X0",
                "ts": "1736481301.000000"
            },
            "client_msg_id": "51e3533b-6fd8-4705-ae34-045ee74db354",
            "text": "ラジオCMはやってましたか？",
            "team": "T02X3XXX1",
            "thread_ts": "1736207085.539649",
            "parent_user_id": "U0XXX00X0",
            "blocks": [
                {
                    "type": "rich_text",
                    "block_id": "tES24",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "ラジオCMはやってましたか？"
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        "channel": "C0000XXX0XX",
        "hidden": True,
        "deleted_ts": "1736481285.683829",
        "event_ts": "1738194975.000100",
        "ts": "1738194975.000100",
        "channel_type": "group"
    }
    event_message_in_thread_deleted = SlackMessageEvent(message_in_thread_deleted)
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

    message2 = {
        "user": "U0XXX00X0",
        "type": "message",
        "ts": "1738209531.687019",
        "client_msg_id": "92e6efd3-b29a-4189-802c-ee1815bb60a4",
        "text": "テストメッセージ",
        "team": "T02X3XXX1",
        "blocks": [
            {
                "type": "rich_text",
                "block_id": "9B7aZ",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [
                            {
                                "type": "text",
                                "text": "テストメッセージ"
                            }
                        ]
                    }
                ]
            }
        ],
        "channel": "C0000XXX0XX",
        "event_ts": "1738209531.687019",
        "channel_type": "group"
    }
    event_message2 = SlackMessageEvent(message2)
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

    message2_deleted = {
        "type": "message",
        "subtype": "message_deleted",
        "previous_message": {
            "user": "U0XXX00X0",
            "type": "message",
            "ts": "1738209531.687019",
            "client_msg_id": "92e6efd3-b29a-4189-802c-ee1815bb60a4",
            "text": "テストメッセージ",
            "team": "T02X3XXX1",
            "blocks": [
                {
                    "type": "rich_text",
                    "block_id": "9B7aZ",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "テストメッセージ"
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        "channel": "C0000XXX0XX",
        "hidden": True,
        "deleted_ts": "1738209531.687019",
        "event_ts": "1738209537.000400",
        "ts": "1738209537.000400",
        "channel_type": "group"
    }
    event_message2_deleted = SlackMessageEvent(message2_deleted)
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
