# メッセージイベント（通常）
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

# 更新イベント（通常）
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

# メッセージイベント（スレッド内）
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

# 更新イベント（スレッド内）
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

# 削除イベント（スレッドのルートメッセージ）
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

# 削除イベント（スレッドの子メッセージ）
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

# メッセージイベント2（通常）
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

# 削除イベント2（非スレッド）
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

# メッセージイベント（ファイル送信）
file_share_message = {
    "text": "<@U0761XXXX6X> 添付したファイルの内容をそのまま復唱してください。",
    "files": [
        {
            "id": "F00XXX0X00X",
            "created": 1738799448,
            "timestamp": 1738799448,
            "name": "無題",
            "title": "無題",
            "mimetype": "text/plain",
            "filetype": "text",
            "pretty_type": "Plain Text",
            "user": "U0XXX00X0",
            "user_team": "T02X3XXX1",
            "editable": True,
            "size": 13,
            "mode": "snippet",
            "is_external": False,
            "external_type": "",
            "is_public": False,
            "public_url_shared": False,
            "display_as_bot": False,
            "username": "",
            "url_private": "https://files.slack.com/files-pri/T02X3XXX1-F00XXX0X00X/______",
            "url_private_download": "https://files.slack.com/files-pri/T02X3XXX1-F00XXX0X00X/download/______",
            "permalink": "https://raccoon-co.slack.com/files/U0XXX00X0/F00XXX0X00X/______",
            "permalink_public": "https://slack-files.com/T02X3XXX1-F00XXX0X00X-f532a53ed6",
            "edit_link": "https://raccoon-co.slack.com/files/U0XXX00X0/F00XXX0X00X/______/edit",
            "preview": "Hello, World!",
            "preview_highlight": "<div class=\"CodeMirror cm-s-default CodeMirrorServer\">\n<div class=\"CodeMirror-code\">\n<div><pre>Hello, World!</pre></div>\n</div>\n</div>\n",
            "lines": 1,
            "lines_more": 0,
            "preview_is_truncated": False,
            "has_rich_preview": False,
            "file_access": "visible"
        }
    ],
    "upload": True,
    "user": "U0XXX00X0",
    "display_as_bot": False,
    "blocks": [
        {
            "type": "rich_text",
            "block_id": "OPmvf",
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
                            "text": " 添付したファイルの内容をそのまま復唱してください。"
                        }
                    ]
                }
            ]
        }
    ],
    "type": "message",
    "ts": "1738799450.020939",
    "client_msg_id": "10b48a02-f312-433c-b4db-b7463e589fb4",
    "channel": "C0000XXX0XX",
    "subtype": "file_share",
    "event_ts": "1738799450.020939",
    "channel_type": "group"
}

# 更新イベント（ファイル送信）
file_share_message_changed = {
    "type": "message",
    "subtype": "message_changed",
    "message": {
        "text": "<@U0761XXXX6X> 添付したファイルの内容をそのまま2回復唱してください。",
        "files": [
            {
                "id": "F00XXX0X00X",
                "created": 1738799448,
                "timestamp": 1738799448,
                "name": "無題",
                "title": "無題",
                "mimetype": "text/plain",
                "filetype": "text",
                "pretty_type": "プレーンテキスト",
                "user": "U0XXX00X0",
                "user_team": "T02X3XXX1",
                "editable": True,
                "size": 13,
                "mode": "snippet",
                "is_external": False,
                "external_type": "",
                "is_public": False,
                "public_url_shared": False,
                "display_as_bot": False,
                "username": "",
                "url_private": "https://files.slack.com/files-pri/T02X3XXX1-F00XXX0X00X/______",
                "url_private_download": "https://files.slack.com/files-pri/T02X3XXX1-F00XXX0X00X/download/______",
                "permalink": "https://raccoon-co.slack.com/files/U0XXX00X0/F00XXX0X00X/______",
                "permalink_public": "https://slack-files.com/T02X3XXX1-F00XXX0X00X-f532a53ed6",
                "edit_link": "https://raccoon-co.slack.com/files/U0XXX00X0/F00XXX0X00X/______/edit",
                "preview": "Hello, World!",
                "preview_highlight": "<div class=\"CodeMirror cm-s-default CodeMirrorServer\">\n<div class=\"CodeMirror-code\">\n<div><pre>Hello, World!</pre></div>\n</div>\n</div>\n",
                "lines": 1,
                "lines_more": 0,
                "preview_is_truncated": False,
                "has_rich_preview": False,
                "file_access": "visible"
            }
        ],
        "upload": True,
        "user": "U0XXX00X0",
        "display_as_bot": False,
        "blocks": [
            {
                "type": "rich_text",
                "block_id": "bUW6c",
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
                                "text": " 添付したファイルの内容をそのまま2回復唱してください。"
                            }
                        ]
                    }
                ]
            }
        ],
        "type": "message",
        "edited": {
            "user": "U0XXX00X0",
            "ts": "1738799457.000000"
        },
        "client_msg_id": "10b48a02-f312-433c-b4db-b7463e589fb4",
        "ts": "1738799450.020939",
        "source_team": "T02X3XXX1",
        "user_team": "T02X3XXX1"
    },
    "previous_message": {
        "text": "<@U0761XXXX6X> 添付したファイルの内容をそのまま復唱してください。",
        "files": [
            {
                "id": "F00XXX0X00X",
                "created": 1738799448,
                "timestamp": 1738799448,
                "name": "無題",
                "title": "無題",
                "mimetype": "text/plain",
                "filetype": "text",
                "pretty_type": "プレーンテキスト",
                "user": "U0XXX00X0",
                "user_team": "T02X3XXX1",
                "editable": True,
                "size": 13,
                "mode": "snippet",
                "is_external": False,
                "external_type": "",
                "is_public": False,
                "public_url_shared": False,
                "display_as_bot": False,
                "username": "",
                "url_private": "https://files.slack.com/files-pri/T02X3XXX1-F00XXX0X00X/______",
                "url_private_download": "https://files.slack.com/files-pri/T02X3XXX1-F00XXX0X00X/download/______",
                "permalink": "https://raccoon-co.slack.com/files/U0XXX00X0/F00XXX0X00X/______",
                "permalink_public": "https://slack-files.com/T02X3XXX1-F00XXX0X00X-f532a53ed6",
                "edit_link": "https://raccoon-co.slack.com/files/U0XXX00X0/F00XXX0X00X/______/edit",
                "preview": "Hello, World!",
                "preview_highlight": "<div class=\"CodeMirror cm-s-default CodeMirrorServer\">\n<div class=\"CodeMirror-code\">\n<div><pre>Hello, World!</pre></div>\n</div>\n</div>\n",
                "lines": 1,
                "lines_more": 0,
                "preview_is_truncated": False,
                "is_starred": False,
                "has_rich_preview": False,
                "file_access": "visible"
            }
        ],
        "upload": True,
        "user": "U0XXX00X0",
        "display_as_bot": False,
        "blocks": [
            {
                "type": "rich_text",
                "block_id": "OPmvf",
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
                                "text": " 添付したファイルの内容をそのまま復唱してください。"
                            }
                        ]
                    }
                ]
            }
        ],
        "type": "message",
        "ts": "1738799450.020939",
        "client_msg_id": "10b48a02-f312-433c-b4db-b7463e589fb4"
    },
    "channel": "C0000XXX0XX",
    "hidden": True,
    "ts": "1738799457.000400",
    "event_ts": "1738799457.000400",
    "channel_type": "group"
}

# 削除イベント（ファイル送信・スレッドのルートメッセージ）
file_share_root_message_deleted = {
    "type": "message",
    "subtype": "message_changed",
    "message": {
        "subtype": "tombstone",
        "text": "This message was deleted.",
        "user": "USLACKBOT",
        "hidden": True,
        "type": "message",
        "thread_ts": "1738799450.020939",
        "reply_count": 1,
        "reply_users_count": 1,
        "latest_reply": "1738799471.594979",
        "reply_users": [
            "U0761XXXX6X"
        ],
        "is_locked": False,
        "ts": "1738799450.020939"
    },
    "previous_message": {
        "text": "<@U0761XXXX6X> 添付したファイルの内容をそのまま2回復唱してください。",
        "files": [
            {
                "id": "F00XXX0X00X",
                "created": 1738799448,
                "timestamp": 1738799448,
                "name": "無題",
                "title": "無題",
                "mimetype": "text/plain",
                "filetype": "text",
                "pretty_type": "プレーンテキスト",
                "user": "U0XXX00X0",
                "user_team": "T02X3XXX1",
                "editable": True,
                "size": 13,
                "mode": "snippet",
                "is_external": False,
                "external_type": "",
                "is_public": False,
                "public_url_shared": False,
                "display_as_bot": False,
                "username": "",
                "url_private": "https://files.slack.com/files-pri/T02X3XXX1-F00XXX0X00X/______",
                "url_private_download": "https://files.slack.com/files-pri/T02X3XXX1-F00XXX0X00X/download/______",
                "permalink": "https://raccoon-co.slack.com/files/U0XXX00X0/F00XXX0X00X/______",
                "permalink_public": "https://slack-files.com/T02X3XXX1-F00XXX0X00X-f532a53ed6",
                "edit_link": "https://raccoon-co.slack.com/files/U0XXX00X0/F00XXX0X00X/______/edit",
                "preview": "Hello, World!",
                "preview_highlight": "<div class=\"CodeMirror cm-s-default CodeMirrorServer\">\n<div class=\"CodeMirror-code\">\n<div><pre>Hello, World!</pre></div>\n</div>\n</div>\n",
                "lines": 1,
                "lines_more": 0,
                "preview_is_truncated": False,
                "is_starred": False,
                "has_rich_preview": False,
                "file_access": "visible"
            }
        ],
        "upload": True,
        "user": "U0XXX00X0",
        "display_as_bot": False,
        "blocks": [
            {
                "type": "rich_text",
                "block_id": "bUW6c",
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
                                "text": " 添付したファイルの内容をそのまま2回復唱してください。"
                            }
                        ]
                    }
                ]
            }
        ],
        "type": "message",
        "ts": "1738799450.020939",
        "edited": {
            "user": "U0XXX00X0",
            "ts": "1738799457.000000"
        },
        "client_msg_id": "10b48a02-f312-433c-b4db-b7463e589fb4",
        "thread_ts": "1738799450.020939",
        "reply_count": 1,
        "reply_users_count": 1,
        "latest_reply": "1738799471.594979",
        "reply_users": [
            "U0761XXXX6X"
        ],
        "is_locked": False,
        "subscribed": True,
        "last_read": "1738799450.020939"
    },
    "channel": "C0000XXX0XX",
    "hidden": True,
    "ts": "1738799648.000600",
    "event_ts": "1738799648.000600",
    "channel_type": "group"
}
