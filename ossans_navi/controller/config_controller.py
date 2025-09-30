import logging
import re
from typing import Any, Optional

from ossans_navi.service.slack_service import SlackService
from ossans_navi.type import ossans_navi_types

logger = logging.getLogger(__name__)


CONFIG_ITEMS = {
    "trusted_bots": "Trusted Bots",
    "allow_responds": "Allow Responds",
    "admin_users": "Admin Users",
    "viewable_private_channels": "Viewable Private Channels",
}

CONFIG_INDEX_BUTTON: list[dict[str, Any]] = [
    {
        "type": "divider"
    },
    {
        "type": "section",
        "text": {
            "type": "plain_text",
            "text": " ",
        },
        "accessory": {
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": "Config Index",
            },
            "action_id": "config.index",
            "value": "config.index",
        }
    }
]


def routing(body: dict[Any, Any], slack_service: SlackService) -> None:
    if not isinstance(channel_id := body["channel"]["id"], str):
        logger.error("Invalid channel_id")
        return
    if not isinstance(thread_ts := body["message"]["thread_ts"], str):
        logger.error("Invalid thread_ts")
        return
    for action in body["actions"]:
        if isinstance(action, dict) and isinstance(action.get("action_id"), str) and isinstance(action.get("value", ""), str):
            if action["action_id"] == "config.index":
                index(slack_service, channel_id, thread_ts)
            else:
                do_action(slack_service, channel_id, thread_ts, action["action_id"], action.get("value"))


def index(slack_service: SlackService, channel_id: str, thread_ts: str) -> None:
    blocks: list[dict[str, Any]] = []
    for (category, label) in CONFIG_ITEMS.items():
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{label}*"
                }
            }
        )
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "list",
                        },
                        "action_id": f"{category}.list",
                        "value": f"{category}.list",
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "add",
                        },
                        "action_id": f"{category}.add_input",
                        "value": f"{category}.add_input",
                    },
                ]
            }
        )
    slack_service.chat_post_message(
        channel=channel_id,
        thread_ts=thread_ts,
        blocks=blocks
    )


def do_action(slack_service: SlackService, channel_id: str, thread_ts: str, action_id: str, value: Optional[str] = None) -> None:
    if (
        match := re.match(
            f"^({'|'.join([category for category in CONFIG_ITEMS])})\\.(list|add_input|add_search|add_execute|remove_execute)$",
            action_id
        )
    ) is None:
        logger.error(f"Invalid action_id: {action_id}")
        return
    category: str = match[1]
    action: str = match[2]
    value = (value.strip() if isinstance(value, str) else None)
    current_config = get_config(slack_service)

    def chat_post_message(blocks: list) -> None:
        slack_service.chat_post_message(
            channel=channel_id,
            thread_ts=thread_ts,
            blocks=[{"type": "divider"}, *blocks, *CONFIG_INDEX_BUTTON],
        )

    label = CONFIG_ITEMS[category]
    if category in ("trusted_bots", "allow_responds", "admin_users"):
        # 現在の設定を取得
        user_ids: list[str] = []
        match category:
            case "trusted_bots":
                user_ids = current_config.trusted_bots
            case "allow_responds":
                user_ids = current_config.allow_responds
            case "admin_users":
                user_ids = current_config.admin_users
        if action == "list":
            now_users = {
                **(
                    {
                        user.user_id: user
                        for user in [slack_service.get_user(user_id) for user_id in user_ids]
                        if user.is_valid
                    }
                ),
            }
            chat_post_message([
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{label}* - list"
                    }
                },
                *[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"<@{user.user_id}>"
                        },
                        "accessory": {
                            "type": "button",
                            "action_id": f"{category}.remove_execute",
                            "text": {
                                "type": "plain_text",
                                "emoji": True,
                                "text": "Remove"
                            },
                            "value": user.user_id,
                            "style": "danger",
                        }
                    }
                    for user in now_users.values()
                ]
            ])
        elif action == "add_input":
            chat_post_message([
                {
                    "dispatch_action": True,
                    "type": "input",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": f"{category}.add_search"
                    },
                    "label": {
                        "type": "plain_text",
                        "text": f"{label} - Search by display name or User ID",
                        "emoji": True
                    }
                }
            ])
        elif action == "add_search":
            if not value:
                logger.error("Search value is empty")
                chat_post_message([
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "Search value is empty"
                        }
                    }
                ])
                return
            users = slack_service.users_list()
            keyword = value.lower()
            users = [
                user for user in users
                if (
                    keyword in user.name.lower()
                    or keyword in user.username.lower()
                    or keyword in user.user_id.lower()
                )
            ]
            match category:
                case "trusted_bots":
                    users = [user for user in users if user.is_bot]
                case "admin_users":
                    users = [user for user in users if not user.is_bot and not user.is_guest]
            chat_post_message([
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Add {label}*"
                    }
                },
                *[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"<@{user.user_id}>"
                        },
                        "accessory": {
                            "type": "button",
                            "action_id": f"{category}.add_execute",
                            "text": {
                                "type": "plain_text",
                                "emoji": True,
                                "text": "Add"
                            },
                            "value": user.user_id,
                            "style": "primary",
                        }
                    }
                    for user in users[:20]
                ]
            ])
        elif action == "add_execute":
            if not isinstance(value, str):
                logger.error(f"Add {label}: value is invalid: {value}")
                chat_post_message([
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"Add {label}: value is invalid: {value}"
                        }
                    }
                ])
                return
            user_ids.append(value)
            store_config(slack_service, current_config, True)
            chat_post_message([
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Added <@{value}> to {label}"
                    }
                }
            ])
        elif action == "remove_execute":
            if not isinstance(value, str):
                logger.error(f"Remove {label}: value is invalid: {value}")
                chat_post_message([
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"Remove {label}: value is invalid: {value}"
                        }
                    }
                ])
                return
            if value not in user_ids:
                logger.error(f"Remove {label}: value is not found: {value}")
                chat_post_message([
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"Remove {label}: value is not found: {value}"
                        }
                    }
                ])
                return
            user_ids.remove(value)
            store_config(slack_service, current_config, True)
            chat_post_message([
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Removed <@{value}> from {label}"
                    }
                }
            ])
    elif category == "viewable_private_channels":
        channel_ids = current_config.viewable_private_channels
        #  現在有効な viewable_private_channels を取得
        if action == "list":
            now_channels = {
                channel.channel_id: channel
                for channel in [
                    slack_service.get_channel(channel_id, True) for channel_id in channel_ids
                ]
                if channel.is_valid and channel.is_private
            }
            chat_post_message([
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{label}* - list"
                    }
                },
                *[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"<#{channel.channel_id}>"
                        },
                        "accessory": {
                            "type": "button",
                            "action_id": f"{category}.remove_execute",
                            "text": {
                                "type": "plain_text",
                                "emoji": True,
                                "text": "Remove"
                            },
                            "value": channel.channel_id,
                            "style": "danger",
                        }
                    }
                    for channel in now_channels.values()
                ]
            ])
        elif action == "add_input":
            chat_post_message([
                {
                    "dispatch_action": True,
                    "type": "input",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": f"{category}.add_execute"
                    },
                    "label": {
                        "type": "plain_text",
                        "text": f"{label} - Enter the channel_id to add",
                        "emoji": True
                    }
                }
            ])
        elif action == "add_execute":
            if not isinstance(value, str):
                logger.error(f"Add {label}: value is invalid: {value}")
                chat_post_message([
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"Add {label}: value is invalid: {value}"
                        }
                    }
                ])
                return
            target_channel = slack_service.get_channel(value, True)
            if not target_channel.is_valid or not target_channel.is_private:
                logger.error(f"Add {label}: value is invalid or is not private: {value}")
                chat_post_message([
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"Add {label}: value is invalid or is not private: {value}"
                        }
                    }
                ])
                return
            channel_ids.append(value)
            store_config(slack_service, current_config, True)
            chat_post_message([
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Addeded <#{value}> to {label}"
                    }
                }
            ])
        elif action == "remove_execute":
            if not isinstance(value, str):
                logger.error(f"Remove {label}: value is invalid: {value}")
                chat_post_message([
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"Remove {label}: value is invalid: {value}"
                        }
                    }
                ])
                return
            if value not in channel_ids:
                logger.error(f"Remove {label}: value is not found: {value}")
                chat_post_message([
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"Remove {label}: value is not found: {value}"
                        }
                    }
                ])
                return
            channel_ids.remove(value)
            store_config(slack_service, current_config, True)
            chat_post_message([
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Removed <#{value}> from {label}"
                    }
                }
            ])


def store_config(slack_service: SlackService, config: ossans_navi_types.OssansNaviConfig, clear_cache: bool = True) -> None:
    slack_service.store_config_dict(config.to_dict(), clear_cache)


def get_config(slack_service: SlackService) -> ossans_navi_types.OssansNaviConfig:
    config_dict = slack_service.get_config_dict()
    return ossans_navi_types.OssansNaviConfig.from_dict(config_dict)
