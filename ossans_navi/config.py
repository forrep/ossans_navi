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

AI_SERVICE_TYPE = AiServiceType(os.environ.get("OSN_AI_SERVICE_TYPE", "gemini"))
OPENAI_API_KEY = os.environ.get("OSN_OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.environ.get("OSN_AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("OSN_AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = "2024-10-21"
GEMINI_API_KEY = os.environ.get("OSN_GEMINI_API_KEY")
AI_MODEL_LOW_COST = None
AI_MODEL_HIGH_QUALITY = None

# AiModelInfo の名前を指定
# 候補:
#   - GEMINI_20_FLASH
#   - GEMINI_25_FLASH
#   - GEMINI_25_FLASH_LITE
#   - GEMINI_25_PRO
#   - GPT_41
#   - GPT_41_MINI
#   - AZURE_GPT_41
#   - AZURE_GPT_41_MINI
match AI_SERVICE_TYPE:
    case AiServiceType.OPENAI:
        AI_MODEL_LOW_COST = os.environ.get("OSN_OPENAI_MODEL_LOW_COST", os.environ.get("OSN_AI_MODEL_LOW_COST", "GPT_41_MINI"))
        AI_MODEL_HIGH_QUALITY = os.environ.get("OSN_OPENAI_MODEL_HIGH_QUALITY", os.environ.get("OSN_AI_MODEL_HIGH_QUALITY", "GPT_41"))
    case AiServiceType.AZURE_OPENAI:
        AI_MODEL_LOW_COST = os.environ.get("OSN_AZURE_OPENAI_MODEL_LOW_COST", os.environ.get("OSN_AI_MODEL_LOW_COST", "AZURE_GPT_41_MINI"))
        AI_MODEL_HIGH_QUALITY = os.environ.get("OSN_AZURE_OPENAI_MODEL_HIGH_QUALITY", os.environ.get("OSN_AI_MODEL_HIGH_QUALITY", "AZURE_GPT_41"))
    case AiServiceType.GEMINI:
        AI_MODEL_LOW_COST = os.environ.get("OSN_GEMINI_MODEL_LOW_COST", os.environ.get("OSN_AI_MODEL_LOW_COST", "GEMINI_20_FLASH"))
        AI_MODEL_HIGH_QUALITY = os.environ.get("OSN_GEMINI_MODEL_HIGH_QUALITY", os.environ.get("OSN_AI_MODEL_HIGH_QUALITY", "GEMINI_25_FLASH"))
    case _:
        raise ValueError(f"Unknown AI service type: {AI_SERVICE_TYPE}")

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
MAX_IMAGE_SIZE = int(os.environ.get("OSN_MAX_IMAGE_SIZE", "3072"))

# refine_slack_searches を同時実行するスレッド数
REFINE_SLACK_SEARCHES_THREADS = 4
# refine_slack_searches の実行回数と実行深度（メンションあり）
REFINE_SLACK_SEARCHES_COUNT_WITH_MENTION = 4
REFINE_SLACK_SEARCHES_DEPTH_WITH_MENTION = 2
# refine_slack_searches の実行回数と実行深度（メンションなし）
REFINE_SLACK_SEARCHES_COUNT_NO_MENTION = 3
REFINE_SLACK_SEARCHES_DEPTH_NO_MENTION = 2

MAX_OUTPUT_TOKENS = 12288

if AI_SERVICE_TYPE == AiServiceType.GEMINI:
    # Gemini の場合は API利用料金が安いのとコンテキストサイズが大きいので入力トークン数を増量
    # 入力する会話コンテキスト（スレッド）の最大トークン数
    MAX_THREAD_TOKENS = 36000
    # refine_slack_searches の1回あたり許容するトークン数
    REFINE_SLACK_SEARCHES_TOKEN = 24000
    # refine_slack_searches の実行回数と実行深度（メンションあり）
    REFINE_SLACK_SEARCHES_COUNT_WITH_MENTION = 6
    # refine_slack_searches の実行回数と実行深度（メンションなし）
    REFINE_SLACK_SEARCHES_COUNT_NO_MENTION = 4
    # lastshot で許容するトークン数
    LASTSHOT_TOKEN_WITH_MENTION = 80000
    # lastshot で許容するトークン数（メンションなし）
    LASTSHOT_TOKEN_NO_MENTION = 40000
    # lastshot で再入力する画像数
    LASTSHOT_INPUT_IMAGE_FILES = 4
    # 映像・音声ファイルを入力する
    LOAD_VIDEO_AUDIO_FILES = True
    # 映像ファイルの FPS、FPS=0.5 で 2秒ごとに1フレーム消費、1フレームあたり 258トークン消費
    VIDEO_FPS = 0.5
else:
    # 入力する会話コンテキスト（スレッド）の最大トークン数
    MAX_THREAD_TOKENS = 12000
    # refine_slack_searches の1回あたり許容するトークン数
    REFINE_SLACK_SEARCHES_TOKEN = 24000
    # lastshot で許容するトークン数
    LASTSHOT_TOKEN_WITH_MENTION = 20000
    # lastshot で許容するトークン数（メンションなし）
    LASTSHOT_TOKEN_NO_MENTION = 10000
    # lastshot で再入力する画像数
    LASTSHOT_INPUT_IMAGE_FILES = 2
    # 映像・音声ファイルを入力する
    LOAD_VIDEO_AUDIO_FILES = False
    # 映像ファイルの FPS、現時点では Gemini のみのサポート
    VIDEO_FPS = 0.5

# 外部URLの取得をするか
LOAD_URL_CONTEXT = GEMINI_API_KEY is not None

# 開発モードのデフォルト値（起動時に --production が渡されると上書きされて本番モードになる）
DEVELOPMENT_MODE = True

# 限られたプライベートチャネルしか閲覧しない安全モード
SAFE_MODE = True

# 静かモード、メンションしていないと反応しないなど
SILENT_MODE = False

# 処理中のリアクション
PROGRESS_REACTION_THINKING = "thinking_face"
PROGRESS_REACTION_SEARCH = "mag"
PROGRESS_REACTION_REFINE = "memo"
PROGRESS_REACTION_LASTSHOT = "speech_balloon"
