import datetime
import time

from jinja2 import Template

from ossans_navi import config
from ossans_navi.service import ai_prompt_assets
from ossans_navi.type.slack_type import SlackMessageEvent


class AiPromptService:
    def __init__(
        self,
        event: SlackMessageEvent,
        assistant_names: list[str],
    ):
        self.event = event
        self.context = {
            "now": datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
            "workspace_name": config.WORKSPACE_NAME,
            "language": config.LANGUAGE,
            "assistant_names": assistant_names,
        }

    def get_lastshot_system_prompt(self) -> str:
        return self.get_prompt(ai_prompt_assets.LASTSHOT_PROMPT)

    def get_prompt(self, template: str) -> str:
        return Template(template).render(
            {
                **self.context,
                "event": {
                    "channel": self.event.channel,
                    "settings": self.event.settings,
                    "is_open_channel": self.event.is_open_channel(),
                }
            }
        )
