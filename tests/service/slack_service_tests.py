import os
import re

import pytest
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

from ossans_navi.service.slack_service import SlackService
from ossans_navi.service.slack_wrapper import SlackWrapper
from ossans_navi.type.slack_type import SlackChannel, SlackUser


def dedent(text, indent: int = 8) -> str:
    return re.sub(f"^ {{{indent}}}", '', text, 0, re.MULTILINE)


@pytest.fixture(scope="session", autouse=True)
def set_env_vers():
    os.environ["OSN_SLACK_APP_TOKEN"] = "app_token_dummy"
    os.environ["OSN_SLACK_USER_TOKEN"] = "user_token_dummy"
    os.environ["OSN_SLACK_BOT_TOKEN"] = "bot_token_dummy"


def test_convert_markdown_to_mrkdwn_title():
    markdown_text = dedent("""
    - list1
        - list2
    * list1
        * list2
    　- list3
    -list10
    # title
    ## title
    ### title
    #### title
    ##### title
    #title
        # title
    some text **strong text** normal text
    some text**strong text**normal text
    some text `strong text` normal text
    some text`strong text`normal text
    link to [example.com](https://example.com/) here
    """, 4).strip()
    assert SlackService.convert_markdown_to_mrkdwn(markdown_text) == dedent("""
    • list1
        • list2
    • list1
        • list2
    　• list3
    -list10
    *# title*
    *## title*
    *### title*
    *#### title*
    ##### title
    #title
        # title
    some text *strong text* normal text
    some text *strong text* normal text
    some text `strong text` normal text
    some text `strong text` normal text
    link to <https://example.com/|example.com> here
    """, 4).strip()


def test_convert_markdown_to_mrkdwn_codeblock():
    markdown_text = dedent("""
    ```markdown
    # title
    ## title
    some text **strong text** normal text
    ```

    ```html
    code block
    ```

    ```1234
    code block
    ```
    some text **strong text** normal text
    ```
    # title
    ## title
    some text **strong text** normal text
    ```
    test
    ```■本日の業務----------
    - テスト1
    - テスト2```
    [test](http://example.com/)
    ```markdown
    - line1
    ```
    """, 4).strip()
    assert SlackService.convert_markdown_to_mrkdwn(markdown_text) == dedent("""
    ```
    # title
    ## title
    some text **strong text** normal text
    ```

    ```
    code block
    ```

    ```
    1234
    code block
    ```
    some text *strong text* normal text
    ```
    # title
    ## title
    some text **strong text** normal text
    ```
    test
    ```
    ■本日の業務----------
    - テスト1
    - テスト2```
    <http://example.com/|test>
    ```
    - line1
    ```
    """, 4).strip()


@pytest.fixture
def slack_service(monkeypatch: pytest.MonkeyPatch):
    def app_dummy(self, *args, **kwargs):
        pass
    monkeypatch.setattr(App, '__init__', app_dummy)
    monkeypatch.setattr(SocketModeHandler, '__init__', app_dummy)
    return SlackService()


def test_get_user(slack_service: SlackService, monkeypatch: pytest.MonkeyPatch):
    def users_info_dummy(self, user: str):
        if user == "U7CAL37X0":
            return {
                "ok": True,
                "user": {
                    "id": "U7CAL37X0",
                    "team_id": "T02L3BLC1",
                    "name": "yamada",
                    "deleted": False,
                    "color": "c386df",
                    "real_name": "山田 太郎",
                    "tz": "Asia/Tokyo",
                    "tz_label": "Japan Standard Time",
                    "tz_offset": 32400,
                    "profile": {
                        "title": "",
                        "phone": "",
                        "skype": "",
                        "real_name": "山田 太郎",
                        "real_name_normalized": "山田 太郎",
                        "display_name": "yamada.taro",
                        "display_name_normalized": "yamada.taro",
                        "fields": None,
                        "status_text": "",
                        "status_emoji": "",
                        "status_emoji_display_info": [],
                        "status_expiration": 0,
                        "avatar_hash": "0f9d312d5edb",
                        "start_date": "2000-03-20",
                        "image_original": "https://avatars.slack-edge.com/2017-10-12/254469866048_0f9d312d5edb4ae3b1c7_original.jpg",
                        "is_custom_image": True,
                        "huddle_state": "default_unset",
                        "huddle_state_expiration_ts": 0,
                        "first_name": "山田",
                        "last_name": "太郎",
                        "image_24": "https://avatars.slack-edge.com/2017-10-12/254469866048_0f9d312d5edb4ae3b1c7_24.jpg",
                        "image_32": "https://avatars.slack-edge.com/2017-10-12/254469866048_0f9d312d5edb4ae3b1c7_32.jpg",
                        "image_48": "https://avatars.slack-edge.com/2017-10-12/254469866048_0f9d312d5edb4ae3b1c7_48.jpg",
                        "image_72": "https://avatars.slack-edge.com/2017-10-12/254469866048_0f9d312d5edb4ae3b1c7_72.jpg",
                        "image_192": "https://avatars.slack-edge.com/2017-10-12/254469866048_0f9d312d5edb4ae3b1c7_192.jpg",
                        "image_512": "https://avatars.slack-edge.com/2017-10-12/254469866048_0f9d312d5edb4ae3b1c7_512.jpg",
                        "image_1024": "https://avatars.slack-edge.com/2017-10-12/254469866048_0f9d312d5edb4ae3b1c7_1024.jpg",
                        "status_text_canonical": "",
                        "team": "T02L3BLC1"
                    },
                    "is_admin": False,
                    "is_owner": False,
                    "is_primary_owner": False,
                    "is_restricted": False,
                    "is_ultra_restricted": False,
                    "is_bot": False,
                    "is_app_user": False,
                    "updated": 1721801012,
                    "is_email_confirmed": True,
                    "who_can_share_contact_card": "EVERYONE"
                }
            }
        if user.startswith("U1"):
            raise SlackApiError("", {
                "ok": False,
                "error": "user_not_found"
            })
        if user.startswith("U2"):
            raise SlackApiError("", {
                "ok": False,
                "error": "user_not_visible"
            })
        if user.startswith("U3"):
            raise SlackApiError("", {
                "ok": False,
                "error": "request_timeout"
            })
        raise ValueError()

    monkeypatch.setattr(SlackWrapper, "users_info", users_info_dummy)

    assert slack_service.get_user("U7CAL37X0") == SlackUser(
        user_id='U7CAL37X0',
        name='山田 太郎',
        username='yamada',
        mention='<@U7CAL37X0>',
        is_bot=False,
        is_guest=False,
        is_admin=False,
        is_valid=True,
    )

    # users_info で user_not_found が返ってくるパターン
    assert slack_service.get_user("U1XXXXXXX") == SlackUser(
        user_id='U1XXXXXXX',
        name='Unknown',
        username='Unknown',
        mention='',
        is_bot=False,
        is_guest=True,
        is_admin=False,
        is_valid=False,
    )

    # users_info で user_not_visible が返ってくるパターン
    assert slack_service.get_user("U2XXXXXXX") == SlackUser(
        user_id='U2XXXXXXX',
        name='Unknown',
        username='Unknown',
        mention='',
        is_bot=False,
        is_guest=True,
        is_admin=False,
        is_valid=False,
    )

    # users_info で request_timeout が返るパターン
    with pytest.raises(SlackApiError):
        slack_service.get_user("U3XXXXXXX")

    # users_info で ValueError() が返るパターン
    with pytest.raises(ValueError):
        slack_service.get_user("RAISE_ERROR_TEST")

    # users_info を None にしても cache から取得できる
    monkeypatch.setattr(SlackWrapper, "users_info", None)
    assert slack_service.get_user("U7CAL37X0") == SlackUser(
        user_id='U7CAL37X0',
        name='山田 太郎',
        username='yamada',
        mention='<@U7CAL37X0>',
        is_bot=False,
        is_guest=False,
        is_admin=False,
        is_valid=True,
    )


def test_get_bot(slack_service: SlackService, monkeypatch: pytest.MonkeyPatch):
    def bots_info_dummy(self, bot: str):
        if bot == "BCL3TC9NW":
            return {
                "ok": True,
                "bot": {
                    "id": "BCL3TC9NW",
                    "deleted": False,
                    "name": "Good || New TeamBuilder",
                    "updated": 1595922010,
                    "app_id": "A0F7XDUAZ",
                    "icons": {
                        "image_36": "https://avatars.slack-edge.com/2018-09-03/428285650437_178ec1ec6d89ec78dd37_36.png",
                        "image_48": "https://avatars.slack-edge.com/2018-09-03/428285650437_178ec1ec6d89ec78dd37_48.png",
                        "image_72": "https://avatars.slack-edge.com/2018-09-03/428285650437_178ec1ec6d89ec78dd37_72.png"
                    }
                }
            }
        if bot.startswith("B1"):
            raise SlackApiError("", {
                "ok": False,
                "error": "bot_not_found"
            })
        if bot.startswith("B2"):
            raise SlackApiError("", {
                "ok": False,
                "error": "request_timeout"
            })
        raise ValueError()

    monkeypatch.setattr(SlackWrapper, "bots_info", bots_info_dummy)

    assert slack_service.get_bot("BCL3TC9NW") == SlackUser(
        user_id="BCL3TC9NW",
        name="Good || New TeamBuilder",
        username="Good || New TeamBuilder",
        mention="<@BCL3TC9NW>",
        is_bot=True,
        is_guest=False,
        is_admin=False,
        is_valid=True,
        bot_id="BCL3TC9NW",
    )

    # bots_info で bot_not_found が返ってくるパターン
    assert slack_service.get_bot("B1XXXXXXX") == SlackUser(
        user_id='B1XXXXXXX',
        name='Unknown Bot',
        username='unknown_bot',
        mention='',
        is_bot=True,
        is_guest=False,
        is_admin=False,
        is_valid=False,
        bot_id="B1XXXXXXX",
    )

    # bots_info で request_timeout が返るパターン
    with pytest.raises(SlackApiError):
        slack_service.get_bot("B2XXXXXXX")

    # bots_info で ValueError() が返るパターン
    with pytest.raises(ValueError):
        slack_service.get_bot("RAISE_ERROR_TEST")

    # bots_info を None にしても cache から取得できる
    monkeypatch.setattr(SlackWrapper, "bots_info", None)
    assert slack_service.get_bot("BCL3TC9NW") == SlackUser(
        user_id="BCL3TC9NW",
        name="Good || New TeamBuilder",
        username="Good || New TeamBuilder",
        mention="<@BCL3TC9NW>",
        is_bot=True,
        is_guest=False,
        is_admin=False,
        is_valid=True,
        bot_id="BCL3TC9NW",
    )


def test_get_conversations_members(slack_service: SlackService, monkeypatch: pytest.MonkeyPatch):
    def conversations_members_dummy(self, channel: str, limit):
        if channel == "C7GGZ82UR":
            return {
                "ok": True,
                "members": [
                    "U02L3BLC5",
                    "U48KQ57L3",
                    "U496LGCUR",
                    "U4BBS3ACF",
                    "U05S968BH70",
                    "U05U7AMDMSR",
                    "U0632DZFHR9",
                    "U066PRQ7CS0",
                    "U06DS98ACSZ",
                    "U06MZAAG012"
                ],
                "response_metadata": {
                    "next_cursor": ""
                }
            }
        if channel.startswith("C1"):
            raise SlackApiError("", {
                "ok": False,
                "error": "channel_not_found"
            })
        if channel.startswith("C2"):
            raise SlackApiError("", {
                "ok": False,
                "error": "request_timeout"
            })
        raise ValueError()

    monkeypatch.setattr(SlackWrapper, "conversations_members", conversations_members_dummy)

    assert slack_service.get_conversations_members("C7GGZ82UR") == [
        "U02L3BLC5",
        "U48KQ57L3",
        "U496LGCUR",
        "U4BBS3ACF",
        "U05S968BH70",
        "U05U7AMDMSR",
        "U0632DZFHR9",
        "U066PRQ7CS0",
        "U06DS98ACSZ",
        "U06MZAAG012"
    ]

    # 存在しないチャネルの場合は空メンバーが返ってくる
    assert slack_service.get_conversations_members("C1XXXXXXX") == []

    # request_timeout が返るパターン
    # ・・・互換性の検証のため、今は例外を送出せずに既存の動作である空データを返す仕様なので代わりのテストを実施
    # with pytest.raises(SlackApiError):
    #     slack_service.get_conversations_members("C2XXXXXXX")
    assert slack_service.get_conversations_members("C2XXXXXXX") == []

    # その他のエラーが発生した場合
    with pytest.raises(ValueError):
        slack_service.get_conversations_members("RAISE_ERROR_TEST")

    # conversations_members を None にしても cache から取得できる
    monkeypatch.setattr(SlackWrapper, "conversations_members", None)
    assert slack_service.get_conversations_members("C7GGZ82UR") == [
        "U02L3BLC5",
        "U48KQ57L3",
        "U496LGCUR",
        "U4BBS3ACF",
        "U05S968BH70",
        "U05U7AMDMSR",
        "U0632DZFHR9",
        "U066PRQ7CS0",
        "U06DS98ACSZ",
        "U06MZAAG012"
    ]


def test_get_presence(slack_service: SlackService, monkeypatch: pytest.MonkeyPatch):
    def users_getPresence_dummy(self, user: str):
        if user == "U7CAL37X0":
            return {
                "ok": True,
                "presence": "active"
            }
        if user == "U4XXXXXXX":
            return {
                "ok": True,
                "presence": "away"
            }
        if user.startswith("U1"):
            raise SlackApiError("", {
                "ok": False,
                "error": "user_not_found"
            })
        if user.startswith("U2"):
            raise SlackApiError("", {
                "ok": False,
                "error": "user_not_visible"
            })
        if user.startswith("U3"):
            raise SlackApiError("", {
                "ok": False,
                "error": "request_timeout"
            })
        raise ValueError()

    monkeypatch.setattr(SlackWrapper, "users_getPresence", users_getPresence_dummy)

    assert slack_service.get_presence("U7CAL37X0")

    assert not slack_service.get_presence("U4XXXXXXX")

    assert not slack_service.get_presence("U1XXXXXXX")

    assert not slack_service.get_presence("U2XXXXXXX")

    with pytest.raises(SlackApiError):
        slack_service.get_presence("U3XXXXXXX")

    # ValueError() が返るパターン
    with pytest.raises(ValueError):
        slack_service.get_presence("RAISE_ERROR_TEST")


def test_get_channels(slack_service: SlackService, monkeypatch: pytest.MonkeyPatch):
    def conversations_list_dummy1(self, limit: int):
        return {
            "ok": True,
            "channels": [
                {
                    "id": "C02R7KJPW3H",
                    "name": "ts_team_2nd",
                    "is_channel": True,
                    "is_group": False,
                    "is_im": False,
                    "is_mpim": False,
                    "is_private": False,
                    "created": 1640160311,
                    "is_archived": False,
                    "is_general": False,
                    "unlinked": 0,
                    "name_normalized": "ts_team_2nd",
                    "is_shared": False,
                    "is_org_shared": False,
                    "is_pending_ext_shared": False,
                    "pending_shared": [],
                    "context_team_id": "T02L3BLC1",
                    "updated": 1640160311852,
                    "parent_conversation": None,
                    "creator": "U7FDACAJD",
                    "is_ext_shared": False,
                    "shared_team_ids": [
                        "T02L3BLC1"
                    ],
                    "pending_connected_team_ids": [],
                    "is_member": False,
                    "topic": {
                        "value": "",
                        "creator": "",
                        "last_set": 0
                    },
                    "purpose": {
                        "value": "開発第2チームでの共有とか相談とかとか",
                        "creator": "U7FDACAJD",
                        "last_set": 1640160311
                    },
                    "previous_names": [],
                    "num_members": 13
                },
                {
                    "id": "C03KM3J23B9",
                    "name": "ダーツの旅_実況",
                    "is_channel": True,
                    "is_group": False,
                    "is_im": False,
                    "is_mpim": False,
                    "is_private": False,
                    "created": 1655421861,
                    "is_archived": False,
                    "is_general": False,
                    "unlinked": 0,
                    "name_normalized": "ﾀﾞｰﾂの旅_実況",
                    "is_shared": False,
                    "is_org_shared": False,
                    "is_pending_ext_shared": False,
                    "pending_shared": [],
                    "context_team_id": "T02L3BLC1",
                    "updated": 1705295424646,
                    "parent_conversation": None,
                    "creator": "U7G0M3J92",
                    "is_ext_shared": False,
                    "shared_team_ids": [
                        "T02L3BLC1"
                    ],
                    "pending_connected_team_ids": [],
                    "is_member": False,
                    "topic": {
                        "value": "",
                        "creator": "",
                        "last_set": 0
                    },
                    "purpose": {
                        "value": "ダーツの旅の状況を、写真とともにリアルタイムであげてもらうチャンネルです。",
                        "creator": "U7G0M3J92",
                        "last_set": 1655421861
                    },
                    "properties": {
                        "canvas": {
                            "file_id": "F06DVMQCZ18",
                            "is_empty": True,
                            "quip_thread_id": "IOR9AAqvG46"
                        }
                    },
                    "previous_names": [],
                    "num_members": 150
                }
            ],
            "response_metadata": {
                "next_cursor": ""
            }
        }

    def conversations_list_dummy2(self, limit: int):
        raise SlackApiError("", {
            "ok": False,
            "error": "request_timeout"
        })

    monkeypatch.setattr(SlackWrapper, "conversations_list", conversations_list_dummy2)
    with pytest.raises(SlackApiError):
        slack_service.get_channels()

    monkeypatch.setattr(SlackWrapper, "conversations_list", conversations_list_dummy1)
    assert slack_service.get_channels() == {
        "ts_team_2nd": {
            "name": "ts_team_2nd",
            "num_members": 13,
            "score": 4.700439718141093,
        },
        "ダーツの旅_実況": {
            "name": "ダーツの旅_実況",
            "num_members": 150,
            "score": 8.228818690495881,
        },
    }

    # キャッシュで取得可能
    monkeypatch.setattr(SlackWrapper, "conversations_list", conversations_list_dummy2)
    assert slack_service.get_channels() == {
        "ts_team_2nd": {
            "name": "ts_team_2nd",
            "num_members": 13,
            "score": 4.700439718141093,
        },
        "ダーツの旅_実況": {
            "name": "ダーツの旅_実況",
            "num_members": 150,
            "score": 8.228818690495881,
        },
    }


def test_get_channel(slack_service: SlackService, monkeypatch: pytest.MonkeyPatch):
    def conversations_info_dummy(self, channel: str):
        if channel == "C7GGZ82UR":
            return {
                "ok": True,
                "channel": {
                    "id": "C7GGZ82UR",
                    "name": "本社3階",
                    "is_channel": True,
                    "is_group": False,
                    "is_im": False,
                    "is_mpim": False,
                    "is_private": False,
                    "created": 1507776105,
                    "is_archived": False,
                    "is_general": False,
                    "unlinked": 0,
                    "name_normalized": "本社3階",
                    "is_shared": False,
                    "is_org_shared": False,
                    "is_pending_ext_shared": False,
                    "pending_shared": [],
                    "context_team_id": "T02L3BLC1",
                    "updated": 1558159185014,
                    "parent_conversation": None,
                    "creator": "U02L3BLC5",
                    "is_ext_shared": False,
                    "shared_team_ids": [
                        "T02L3BLC1"
                    ],
                    "pending_connected_team_ids": [],
                    "is_member": False,
                    "topic": {
                        "value": "3Fのメンバーだけに告知したい場合はこちら",
                        "creator": "U02L3BLC5",
                        "last_set": 1507776774
                    },
                    "purpose": {
                        "value": "3Fのメンバーにお知らせするチャンネル",
                        "creator": "U02L3BLC5",
                        "last_set": 1507776106
                    },
                    "previous_names": []
                }
            }
        if channel.startswith("C1"):
            raise SlackApiError("", {
                "ok": False,
                "error": "channel_not_found"
            })
        if channel.startswith("C2"):
            raise SlackApiError("", {
                "ok": False,
                "error": "request_timeout"
            })
        raise ValueError()

    monkeypatch.setattr(SlackWrapper, "conversations_info", conversations_info_dummy)

    assert slack_service.get_channel("C7GGZ82UR") == SlackChannel(
        channel_id="C7GGZ82UR",
        name="本社3階",
        topic="3Fのメンバーだけに告知したい場合はこちら",
        purpose="3Fのメンバーにお知らせするチャンネル",
        is_public=True,
        is_private=False,
        is_im=False,
        is_mpim=False,
        is_valid=True,
    )

    assert slack_service.get_channel("C1XXXXXXX") == SlackChannel(
        channel_id="C1XXXXXXX",
        name="Unknown",
        topic="",
        purpose="",
        is_public=False,
        is_private=False,
        is_im=False,
        is_mpim=False,
        is_valid=False,
    )

    # request_timeout が返るパターン
    # ・・・互換性の検証のため、今は例外を送出せずに既存の動作である空データを返す仕様なので代わりのテストを実施
    # with pytest.raises(SlackApiError):
    #     slack_service.get_channel("C2XXXXXXX")
    assert slack_service.get_channel("C2XXXXXXX") == SlackChannel(
        channel_id="C2XXXXXXX",
        name="Unknown",
        topic="",
        purpose="",
        is_public=False,
        is_private=False,
        is_im=False,
        is_mpim=False,
        is_valid=False,
    )
