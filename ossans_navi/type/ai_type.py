from enum import Enum, auto
from typing import Optional

from pydantic import BaseModel, Field

from ossans_navi import config


class AiPromptSlackMessageAttachment(BaseModel):
    title: str
    text: str
    link: str
    name: Optional[str] = Field(default=None, init=False)
    user_id: Optional[str] = Field(default=None, init=False)


class AiPromptSlackMessageFile(BaseModel):
    title: str
    link: str
    description: Optional[str] = Field(default=None, init=False)
    text: Optional[str] = Field(default=None, init=False)


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


class AiServiceType(Enum):
    OPENAI = auto()
    AZURE_OPENAI = auto()
    GEMINI = auto()


class AiModelInfo(Enum):
    GEMINI_20_FLASH = ("gemini-2.0-flash", AiServiceType.GEMINI, 0.10, 0.40)
    GEMINI_25_FLASH = ("gemini-2.5-flash-preview-09-2025", AiServiceType.GEMINI, 0.30, 2.50)
    GEMINI_25_FLASH_LITE = ("gemini-2.5-flash-lite-preview-09-2025", AiServiceType.GEMINI, 0.10, 0.40)
    GEMINI_25_PRO = ("gemini-2.5-pro", AiServiceType.GEMINI, 1.25, 10.00)
    GPT_41 = ("gpt-4.1", AiServiceType.OPENAI, 2.00, 8.00)
    GPT_41_MINI = ("gpt-4.1-mini", AiServiceType.OPENAI, 1.10, 4.40)
    AZURE_GPT_41 = ("gpt-4.1", AiServiceType.AZURE_OPENAI, 2.00, 8.00)
    AZURE_GPT_41_MINI = ("gpt-4.1-mini", AiServiceType.AZURE_OPENAI, 1.10, 4.40)

    def __init__(self, model_name: str, ai_service_type: AiServiceType, cost_in: float, cost_out: float):
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


class AiModelsUsage(BaseModel):
    low_cost: AiModelUsage
    high_quality: AiModelUsage
    gemini_25_flash_lite: Optional[AiModelUsage]
    models: list[AiModelUsage] = Field(default_factory=list, init=False)

    @staticmethod
    def new() -> 'AiModelsUsage':
        try:
            models_usage = AiModelsUsage(
                low_cost=AiModelUsage(model=AiModelInfo[config.AI_MODEL_LOW_COST]),
                high_quality=AiModelUsage(model=AiModelInfo[config.AI_MODEL_HIGH_QUALITY]),
                gemini_25_flash_lite=(AiModelUsage(model=AiModelInfo.GEMINI_25_FLASH_LITE) if config.GEMINI_API_KEY else None),
            )
        except KeyError as e:
            raise ValueError(f"Invalid AI model name in configuration: {e}") from e
        models_usage.models.append(models_usage.low_cost)
        models_usage.models.append(models_usage.high_quality)
        if models_usage.gemini_25_flash_lite:
            models_usage.models.append(models_usage.gemini_25_flash_lite)
        return models_usage

    def get_total_cost(self) -> float:
        return sum([model.get_total_cost() for model in self.models])
