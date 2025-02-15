import os
from enum import Enum

# HTTPプロキシ設定
if "OSN_HTTPS_PROXY" in os.environ:
    os.environ["HTTPS_PROXY"] = os.environ["OSN_HTTPS_PROXY"]


class AiServiceType(Enum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    GEMINI = "gemini"


# 必須設定項目が設定されているかのチェックと取得
SLACK_APP_TOKEN = os.environ["OSN_SLACK_APP_TOKEN"]
SLACK_BOT_TOKEN = os.environ["OSN_SLACK_BOT_TOKEN"]
SLACK_USER_TOKEN = os.environ["OSN_SLACK_USER_TOKEN"]

AI_SERVICE_TYPE = AiServiceType(os.environ["OSN_AI_SERVICE_TYPE"])
match AI_SERVICE_TYPE:
    case AiServiceType.OPENAI:
        OPENAI_API_KEY = os.environ["OSN_OPENAI_API_KEY"]
        OPENAI_MODEL_LOW_COST = os.environ.get("OSN_OPENAI_MODEL_LOW_COST", "gpt-4o-mini")
        OPENAI_MODEL_LOW_COST_IN = float(os.environ.get("OSN_OPENAI_MODEL_LOW_COST_IN", "0"))
        OPENAI_MODEL_LOW_COST_OUT = float(os.environ.get("OSN_OPENAI_MODEL_LOW_COST_OUT", "0"))
        OPENAI_MODEL_HIGH_QUALITY = os.environ.get("OSN_OPENAI_MODEL_HIGH_QUALITY", "gpt-4o")
        OPENAI_MODEL_HIGH_QUALITY_IN = float(os.environ.get("OSN_OPENAI_MODEL_HIGH_QUALITY_IN", "0"))
        OPENAI_MODEL_HIGH_QUALITY_OUT = float(os.environ.get("OSN_OPENAI_MODEL_HIGH_QUALITY_OUT", "0"))
    case AiServiceType.AZURE_OPENAI:
        AZURE_OPENAI_API_KEY = os.environ["OSN_AZURE_OPENAI_API_KEY"]
        AZURE_OPENAI_ENDPOINT = os.environ["OSN_AZURE_OPENAI_ENDPOINT"]
        AZURE_OPENAI_API_VERSION = "2024-06-01"
        AZURE_OPENAI_MODEL_LOW_COST = os.environ.get("OSN_AZURE_OPENAI_MODEL_LOW_COST", "gpt-4o-mini")
        AZURE_OPENAI_MODEL_LOW_COST_IN = float(os.environ.get("OSN_AZURE_OPENAI_MODEL_LOW_COST_IN", "0"))
        AZURE_OPENAI_MODEL_LOW_COST_OUT = float(os.environ.get("OSN_AZURE_OPENAI_MODEL_LOW_COST_OUT", "0"))
        AZURE_OPENAI_MODEL_HIGH_QUALITY = os.environ.get("OSN_AZURE_OPENAI_MODEL_HIGH_QUALITY", "gpt-4o")
        AZURE_OPENAI_MODEL_HIGH_QUALITY_IN = float(os.environ.get("OSN_AZURE_OPENAI_MODEL_HIGH_QUALITY_IN", "0"))
        AZURE_OPENAI_MODEL_HIGH_QUALITY_OUT = float(os.environ.get("OSN_AZURE_OPENAI_MODEL_HIGH_QUALITY_OUT", "0"))
    case AiServiceType.GEMINI:
        GEMINI_API_KEY = os.environ["OSN_GEMINI_API_KEY"]
        GEMINI_MODEL_LOW_COST = os.environ.get("OSN_GEMINI_MODEL_LOW_COST", "gemini-2.0-flash")
        GEMINI_MODEL_LOW_COST_IN = float(os.environ.get("OSN_GEMINI_MODEL_LOW_COST_IN", "0"))
        GEMINI_MODEL_LOW_COST_OUT = float(os.environ.get("OSN_GEMINI_MODEL_LOW_COST_OUT", "0"))
        GEMINI_MODEL_HIGH_QUALITY = os.environ.get("OSN_GEMINI_MODEL_HIGH_QUALITY", "gemini-2.0-flash")
        GEMINI_MODEL_HIGH_QUALITY_IN = float(os.environ.get("OSN_GEMINI_MODEL_HIGH_QUALITY_IN", "0"))
        GEMINI_MODEL_HIGH_QUALITY_OUT = float(os.environ.get("OSN_GEMINI_MODEL_HIGH_QUALITY_OUT", "0"))

WORKSPACE_NAME = v if (v := os.environ.get("OSN_WORKSPACE_NAME")) else "company"
ASSISTANT_NAMES = v.split(r',') if (v := os.environ.get("OSN_ASSISTANT_NAMES")) else ["assistant"]
LANGUAGE = os.environ.get("OSN_LANGUAGE", "user's language")

# 開発用設定
DEVELOPERS = v.split(r',') if (v := os.environ.get("OSN_DEVELOPERS")) else []
# 開発用のチャネル
DEVELOPMENT_CHANNELS = v.split(r',') if (v := os.environ.get("OSN_DEVELOPMENT_CHANNELS")) else []
# OssansNavi の投稿したデータを保持するチャネル
RESPONSE_LOGGING_CHANNEL = os.environ.get("OSN_RESPONSE_LOGGING_CHANNEL")

# 読み込む画像の一辺の最大サイズ、これを超える場合は縮小する
MAX_IMAGE_SIZE = int(os.environ.get("OSN_MAX_IMAGE_SIZE", "2304"))

# 入力する会話コンテキスト（スレッド）の最大トークン数
MAX_CONVERSATION_TOKENS = 8000

# request_refine_slack_searches を同時実行するスレッド数
REQUEST_REFINE_SLACK_SEARCHES_THREADS = 4
# request_refine_slack_searches の実行回数と実行深度（メンションあり）
REQUEST_REFINE_SLACK_SEARCHES_COUNT_WITH_MENTION = 4
REQUEST_REFINE_SLACK_SEARCHES_DEPTH_WITH_MENTION = 2
# request_refine_slack_searches の実行回数と実行深度（メンションなし）
REQUEST_REFINE_SLACK_SEARCHES_COUNT_NO_MENTION = 3
REQUEST_REFINE_SLACK_SEARCHES_DEPTH_NO_MENTION = 2
# request_refine_slack_searches の1回あたり許容するトークン数
REQUEST_REFINE_SLACK_SEARCHES_TOKEN = 30000 if AI_SERVICE_TYPE == AiServiceType.GEMINI else 24000

# request_lastshot で許容するトークン数
REQUEST_LASTSHOT_TOKEN_WITH_MENTION = 40000 if AI_SERVICE_TYPE == AiServiceType.GEMINI else 20000
# request_lastshot で許容するトークン数（メンションなし）
REQUEST_LASTSHOT_TOKEN_NO_MENTION = 20000 if AI_SERVICE_TYPE == AiServiceType.GEMINI else 10000

# 開発モード（デフォルトは開発モード、起動時に --production が渡されると本番モードになる）
DEVELOPMENT_MODE = True

# 限られたプライベートチャネルしか閲覧しない安全モード
SAFE_MODE = True

# 静かモード、メンションしていないと反応しないなど
SILENT_MODE = False

# メッセージに返信する必要がない場合のリアクション対応表
SLACK_REACTIONS = {
    "question": None,                           # 質問
    "request": None,                            # 依頼
    "report": "spiral_note_pad",                # 報告
    "advice": "teacher",                        # アドバイス
    "agreement": "smile",                       # 同意
    "empathy": "relaxed",                       # 共感
    "confirmation": "wink",                     # 確認
    "admiration": "thumbsup",                   # 感嘆
    "disappointment": "disappointed_relieved",  # 失望
    "task_list": "pencil",                      # タスク
    "other": None,
}
