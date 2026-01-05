from enum import Enum, auto
from typing import Any, Optional

from pydantic import BaseModel, Field

from ossans_navi import config


class AiPromptSlackMessageAttachment(BaseModel):
    title: str
    text: str
    link: str
    name: Optional[str] = Field(default=None, init=False)
    user_id: Optional[str] = Field(default=None, init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "text": self.text,
            "link": self.link,
            **({"name": self.name} if self.name is not None else {}),
            **({"user_id": self.user_id} if self.user_id is not None else {}),
        }


class AiPromptSlackMessageFile(BaseModel):
    title: str
    link: str
    description: Optional[str] = Field(default=None, init=False)
    text: Optional[str] = Field(default=None, init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "link": self.link,
            **({"description": self.description} if self.description is not None else {}),
            **({"text": self.text} if self.text is not None else {}),
        }


class AiPromptSlackMessage(BaseModel):
    timestamp: str
    name: str
    user_id: str
    content: str
    permalink: str
    mention_to: Optional[str] = Field(default=None)
    attachments: list[AiPromptSlackMessageAttachment] = Field(default_factory=list, init=False)
    files: list[AiPromptSlackMessageFile] = Field(default_factory=list, init=False)
    reactions: list[str] = Field(default_factory=list, init=False)
    channel: Optional[str] = Field(default=None, init=False)
    root_message: Optional["AiPromptSlackMessage"] = Field(default=None, init=False)
    replies: list["AiPromptSlackMessage"] = Field(default_factory=list, init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "name": self.name,
            "user_id": self.user_id,
            "content": self.content,
            "permalink": self.permalink,
            **({"mention_to": self.mention_to} if self.mention_to is not None else {}),
            **({"attachments": [attachment.to_dict() for attachment in self.attachments]} if len(self.attachments) > 0 else {}),
            **({"files": [file.to_dict() for file in self.files]} if len(self.files) > 0 else {}),
            **({"reactions": self.reactions} if len(self.reactions) > 0 else {}),
            **({"channel": self.channel} if self.channel is not None else {}),
            **({"root_message": self.root_message.to_dict()} if self.root_message is not None else {}),
            **({"replies": [reply.to_dict() for reply in self.replies]} if len(self.replies) > 0 else {}),
        }


class AiServiceType(Enum):
    OPENAI = auto()
    AZURE_OPENAI = auto()
    GEMINI = auto()


class AiModelInfo(Enum):
    GEMINI_20_FLASH = ("gemini-2.0-flash", AiServiceType.GEMINI, 0.10, 0.40)
    GEMINI_25_FLASH = ("gemini-2.5-flash-preview-09-2025", AiServiceType.GEMINI, 0.30, 2.50)
    GEMINI_25_FLASH_LITE = ("gemini-2.5-flash-lite-preview-09-2025", AiServiceType.GEMINI, 0.10, 0.40)
    GEMINI_25_PRO = ("gemini-2.5-pro", AiServiceType.GEMINI, 1.25, 10.00)
    GEMINI_30_FLASH = ("gemini-3-flash-preview", AiServiceType.GEMINI, 0.50, 3.00)
    GPT_41 = ("gpt-4.1", AiServiceType.OPENAI, 2.00, 8.00)
    GPT_41_MINI = ("gpt-4.1-mini", AiServiceType.OPENAI, 1.10, 4.40)
    GPT_5_NANO = ("gpt-5-nano", AiServiceType.OPENAI, 0.05, 0.40)
    GPT_5_MINI = ("gpt-5-mini", AiServiceType.OPENAI, 0.25, 2.00)
    AZURE_GPT_41 = ("gpt-4.1", AiServiceType.AZURE_OPENAI, 2.00, 8.00)
    AZURE_GPT_41_MINI = ("gpt-4.1-mini", AiServiceType.AZURE_OPENAI, 1.10, 4.40)
    AZURE_GPT_5_NANO = ("gpt-5-nano", AiServiceType.AZURE_OPENAI, 0.05, 0.40)
    AZURE_GPT_5_MINI = ("gpt-5-mini", AiServiceType.AZURE_OPENAI, 0.25, 2.00)

    def __init__(
        self,
        model_name: str,
        ai_service_type: AiServiceType,
        cost_in: float,
        cost_out: float,
    ):
        self.model_name = model_name
        self.ai_service_type = ai_service_type
        self.cost_in = cost_in
        """入力コスト（$ / 1,000,000 tokens）"""
        self.cost_out = cost_out
        """出力コスト（$ / 1,000,000 tokens）"""


class AiModelUsage(BaseModel):
    model: AiModelInfo
    tokens_in: int = Field(default=0, init=False)
    tokens_out: int = Field(default=0, init=False)

    @property
    def model_name(self) -> str:
        return self.model.model_name

    @property
    def ai_service_type(self) -> AiServiceType:
        return self.model.ai_service_type

    @property
    def cost_in(self) -> float:
        return self.model.cost_in

    @property
    def cost_out(self) -> float:
        return self.model.cost_out

    def get_total_cost(self) -> float:
        return self.cost_in * self.tokens_in / 1_000_000 + self.cost_out * self.tokens_out / 1_000_000


class AiModelType(Enum):
    LOW_COST = auto()
    HIGH_QUALITY = auto()


class AiModelsUsage(BaseModel):
    models: list[AiModelUsage] = Field(default_factory=list, init=False)
    models_map: dict[str, AiModelUsage] = Field(default_factory=dict, init=False)

    @staticmethod
    def new() -> 'AiModelsUsage':
        models_usage = AiModelsUsage()
        models_usage.models_map.update(
            {
                ai_model_info.name: AiModelUsage(model=ai_model_info)
                for ai_model_info in AiModelInfo
                if (
                    (ai_model_info.ai_service_type == AiServiceType.GEMINI and config.GEMINI_API_KEY)
                    or (ai_model_info.ai_service_type == AiServiceType.OPENAI and config.OPENAI_API_KEY)
                    or (ai_model_info.ai_service_type == AiServiceType.AZURE_OPENAI and config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_ENDPOINT)
                )
            }
        )
        models_usage.models.extend(models_usage.models_map.values())
        if (
            config.AI_MODEL_LOW_COST not in models_usage.models_map
            or config.AI_MODEL_HIGH_QUALITY not in models_usage.models_map
        ):
            raise ValueError(f"Invalid AI model configuration: LOW_COST={config.AI_MODEL_LOW_COST}, HIGH_QUALITY={config.AI_MODEL_HIGH_QUALITY}")
        models_usage.models_map[AiModelType.LOW_COST.name] = models_usage.models_map[config.AI_MODEL_LOW_COST]
        models_usage.models_map[AiModelType.HIGH_QUALITY.name] = models_usage.models_map[config.AI_MODEL_HIGH_QUALITY]

        # モデル名が正常に設定されているかの確認
        for name in dir(config):
            if not name.startswith("MODEL_FOR_"):
                continue
            if (value := getattr(config, name)) in models_usage.models_map:
                continue
            raise ValueError(f"Invalid AI model configuration for {name}: {value}")

        return models_usage

    def get_total_cost(self) -> float:
        return sum([model.get_total_cost() for model in self.models])

    def get_model(self, name: str) -> Optional[AiModelUsage]:
        return self.models_map.get(name)
