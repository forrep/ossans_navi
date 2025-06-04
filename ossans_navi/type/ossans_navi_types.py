import dataclasses


@dataclasses.dataclass
class Image:
    data: bytes
    mime_type: str

    @property
    def extension(self) -> str:
        if self.mime_type == "image/png":
            return ".png"
        elif self.mime_type == "image/jpeg":
            return ".jpg"
        elif self.mime_type == "image/gif":
            return ".gif"
        else:
            return ""


@dataclasses.dataclass
class OssansNaviConfig:
    trusted_bots: list[str] = dataclasses.field(default_factory=list, init=False)
    allow_responds: list[str] = dataclasses.field(default_factory=list, init=False)
    admin_users: list[str] = dataclasses.field(default_factory=list, init=False)
    viewable_private_channels: list[str] = dataclasses.field(default_factory=list, init=False)

    @staticmethod
    def from_dict(settings_dict: dict) -> 'OssansNaviConfig':
        settings = OssansNaviConfig()
        if settings_dict.get("type") == "config":
            if isinstance(settings_dict.get("trusted_bots"), list):
                settings.trusted_bots.extend([v for v in settings_dict["trusted_bots"] if isinstance(v, str)])
            if isinstance(settings_dict.get("allow_responds"), list):
                settings.allow_responds.extend([v for v in settings_dict["allow_responds"] if isinstance(v, str)])
            if isinstance(settings_dict.get("admin_users"), list):
                settings.admin_users.extend([v for v in settings_dict["admin_users"] if isinstance(v, str)])
            if isinstance(settings_dict.get("viewable_private_channels"), list):
                settings.viewable_private_channels.extend([v for v in settings_dict["viewable_private_channels"] if isinstance(v, str)])
        return settings

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class LastshotResponse:
    text: str
    images: list[Image]
