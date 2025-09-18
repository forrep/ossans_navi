import datetime
import time
from typing import Any, Optional

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

    CLASSIFY_SCHEMA = ai_prompt_assets.CLASSIFY_SCHEMA
    IMAGE_DESCRIPTION_SCHEMA = ai_prompt_assets.IMAGE_DESCRIPTION_SCHEMA
    SLACK_SEARCH_WORD_SCHEMA = ai_prompt_assets.SLACK_SEARCH_WORD_SCHEMA
    REFINE_SLACK_SEARCHES_SCHEMA = ai_prompt_assets.REFINE_SLACK_SEARCHES_SCHEMA
    QUALITY_CHECK_SCHEMA = ai_prompt_assets.QUALITY_CHECK_SCHEMA

    def classify_prompt(self) -> str:
        return self._prompt(
            ai_prompt_assets.CLASSIFY_PROMPT,
            {
                "schema": {
                    "user_intentions_type": ai_prompt_assets.USER_INTENTIONS_TYPES,
                    "who_to_talk_to": ai_prompt_assets.WHO_TO_TALK_TO_TYPES,
                    "user_emotions": ai_prompt_assets.USER_EMOTIONS,
                    "required_knowledge_types": ai_prompt_assets.REQUIRED_KNOWLEDGE_TYPES,
                },
            },
        )

    def image_description_prompt(self) -> str:
        return self._prompt(ai_prompt_assets.IMAGE_DESCRIPTION_PROMPT)

    def slack_search_word_prompt(self) -> str:
        return self._prompt(ai_prompt_assets.SLACK_SEARCH_WORD_PROMPT)

    def refine_slack_searches_prompt(self) -> str:
        return self._prompt(ai_prompt_assets.REFINE_SLACK_SEARCHES_PROMPT)

    def lastshot_prompt(self, has_rag_info: bool) -> str:
        return self._prompt(ai_prompt_assets.LASTSHOT_PROMPT, {"has_rag_info": has_rag_info})

    def quality_check_prompt(self, response_message: str) -> str:
        return self._prompt(
            ai_prompt_assets.QUALITY_CHECK_PROMPT,
            {"response_message": response_message},
        )

    def _prompt(self, template: str, extra: Optional[dict[str, Any]] = None) -> str:
        t: Template = Template(template)
        return t.render(
            {
                **self.context,
                "event": {
                    "channel": self.event.channel,
                    "settings": self.event.settings,
                    "is_mention": self.event.is_mention,
                    "is_open_channel": self.event.is_open_channel(),
                    "has_image_video_audio": self.event.has_image_video_audio,
                },
                **(extra or {}),
            }
        ).strip()
