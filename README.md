# OssansNavi
OssansNavi はコミュニケーションのハブとなる **人を繋ぐ AI**  です
宛先のない質問へ「それに詳しいのは○○さんですね、@○○ 回答お願いします」というパスをしてくれます

Slack ワークスペースを情報源とするので、事前に AI を学習させる必要はなく起動するだけで精度の高い応答が可能です
必要なのは、Slack ワークスペースと Gemini/OpenAI/Azure OpenAI いずれかのAPIキーを用意してから OssansNavi アプリを起動するだけです

## 細かい仕様
- パブリックチャネルの情報を利用します、プライベートチャネルはデフォルトでは参照しません（特定のプライベートチャネルを参照する設定あり）
- ゲストやコネクトのユーザーには応答しません
- WebSocket で動作します、外部からアクセス可能な環境は必要ありません
- OssansNavi のボットユーザーが参加しているチャネルや DM で応答します
- OssansNavi をメンションすると確実に応答してくれます
- OssansNavi をメンションしなくても応答可能ならば応答します
  - まずは様々なチャネルに OssansNavi のボットユーザーを追加してみることをオススメします
- OssansNavi が質問に回答できない場合は詳しい人にメンションすることもあります。メンションされた方は広い心で対応しましょう
- 応答に 1~3分はかかるので相手を人間だと思って気長にお待ちください
- 安価な LLM モデルと高精度のモデルを使い分けることで、LLM の API 使用料を節約します

OssansNavi について詳しくは以下の記事をご覧ください

- [RAGにベクトルDBは必要ない！DBも不要で運用めちゃ楽な RAG Chatbot を作った話](https://speakerdeck.com/forrep/rag-does-not-need-a-vector-db)
- [社内用AIアシスタント「おっさんずナビ」を作った話、そして人間らしく振る舞う重要性を認識した話](https://techblog.raccoon.ne.jp/archives/1719796918.html)

## アプリの利用方法
OssansNavi は Slack Marketplace で配布していません。

ご利用の Slack ワークスペースで新規アプリを追加して、その認証キーを設定した OssansNavi バックエンドアプリ（当リポジトリ）を起動することで利用できます。

具体的なインストール手順は以下に従ってください。

### 1. ワークスペースに Slack アプリを追加
https://api.slack.com/apps へアクセスして、以下の手順で Slack アプリを追加します

- "Create New App" ボタンを押して、"Create an app" ダイアログで "From a manifest" を選択します
- "Pick a workspace to develop your app" ダイアログで対象のワークスペースを選択します
- "Create app from manifest" ダイアログで JSON を選択して、[manifest.json](assets/manifest.json) の内容を貼り付けます
  - アプリ名を変更するには "display_information" > "name" と "features" > "bot_user" > "display_name" を任意の名称としてください （※後から変更可能）
- "Review summary & create your app" で "Create" を選択します

### 2. Slack アプリの設定
https://api.slack.com/apps で先ほど追加したアプリを選択して設定します

- "Basic Information" > "App-Level Tokens" > "Generate Token and Scopes" を選択して以下の手順でトークンを発行します
  1. "Token Name" に `app_token` を入力します
  2. "Add Scope" で `connections:write` を追加します
  3. "Generate" ボタンで生成します
  4. 発行された `xapp-...` をメモします ※後で再表示可能
- "Basic Information" > "Display Information" で任意のアプリアイコンと名称を登録します
  - デフォルトアイコンは [ossans_navi.png](assets/ossans_navi.png) です
- "Basic Information" 画面下部の "Save Changes" で変更内容を保存します
- "App Home" > "Show Tabs" で "Allow users to send Slash commands and messages from the messages tab" のチェックを入れます
  - このチェックを入れることでアプリとの DM ができるようになります
- "Install App" で "Install to {workspace_name}" を選択します
  1. "{app_name} is requesting permission to access the {workspace_name} Slack workspace" が表示されたら内容を確認して "Allow" を選択します
  2. "User OAuth Token", "Bot User OAuth Token" をメモします ※後で再表示可能

### 3. バックエンドアプリの設定
バックエンドアプリの設定を記述した `.env` ファイルを用意します。

以下の内容を `.env` という名前で保存してください。`OSN_SLACK_APP_TOKEN` `OSN_SLACK_BOT_TOKEN` `OSN_SLACK_USER_TOKEN` には、前の手順で発行したトークンをそれぞれ設定します。
LLM には Gemini, OpenAI, Azure OpenAI を利用可能です。

```properties
# Slack トークン
OSN_SLACK_APP_TOKEN=xapp-...
OSN_SLACK_BOT_TOKEN=xoxb-...
OSN_SLACK_USER_TOKEN=xoxp-...

# Gemini API キー（Gemini を利用する場合は設定する）
OSN_GEMINI_API_KEY=AIz...
# OpenAI API キー（OpenAI を利用する場合は設定する）
OSN_OPENAI_API_KEY=684...
# Azure OpenAI API キー及びエンドポイントのドメイン（Azure OpenAI を利用する場合は設定する）
OSN_AZURE_OPENAI_API_KEY=684...
OSN_AZURE_OPENAI_ENDPOINT=https://*.openai.azure.com/
```

OSN_AI_MODEL_LOW_COST/OSN_AI_MODEL_HIGH_QUALITY で利用する LLM モデルを設定します。以下が許容される設定値です。

- GEMINI_20_FLASH （デフォルトの低コストモデル）
- GEMINI_25_FLASH （デフォルトの高クオリティモデル）
- GEMINI_25_FLASH_LITE
- GEMINI_25_PRO
- GPT_41
- GPT_41_MINI
- AZURE_GPT_41 （デプロイ名は gpt-4.1 固定）
- AZURE_GPT_41_MINI （デプロイ名は gpt-4.1-mini 固定）

```properties
# 低コストモデル（デフォルト: GEMINI_20_FLASH）
OSN_AI_MODEL_LOW_COST=GEMINI_20_FLASH
# 高クオリティモデル（デフォルト: GEMINI_25_FLASH）
OSN_AI_MODEL_HIGH_QUALITY=GEMINI_25_FLASH
```

```properties
# -- アプリの設定 --
# Slack ワークスペースの名称、典型的には社名等を指定、システムプロンプトで入力されて OssansNavi が自身の稼働する環境を認識するために利用
OSN_WORKSPACE_NAME=ABC株式会社
# OssansNavi の呼称、Slack ワークスペース内で OssansNavi がなんと呼ばれるかを指定（カンマ区切りで複数可）、OssansNavi が自身への呼びかけや文脈を理解するために利用する
OSN_ASSISTANT_NAMES=OssansNavi,おっさんずナビ,おっさん
# 言語、OssansNavi が Slack ワークスペース内の検索及び応答に利用する言語、省略時はユーザーの言語になるべく合わせて応答
OSN_LANGUAGE=Japanese
# ログレベル
OSN_LOG_LEVEL=INFO
# OssansNavi の応答をロギングするチャネル ※省略可 例: {"channel": "CXXXXXXXX", "thread_ts", "cost": 0.04}
# OSN_RESPONSE_LOGGING_CHANNEL=
# 読み込む画像の一辺の最大サイズ、これを超える場合は縮小する
# OSN_MAX_IMAGE_SIZE=2304
```

### 4. バックエンドアプリの起動
```
docker run --env-file .env ghcr.io/forrep/ossans_navi:latest
```

起動に成功すると以下のメッセージが表示されます。（※ OSN_LOG_LEVEL=INFO 以上が必要）

```
slack_bolt(265) - INFO - Starting to receive messages from a new connection 
```

まずは Slack の画面から OssansNavi アプリを開いて DM で話しかけてみましょう。
質問内容にもよりますが 1~3分ほどで応答してくれるはずです。

### バックエンドアプリを終了
`docker stop` や `kill` でプロセスに TERM シグナルを送信すると Graceful Shutdown します。
Graceful Shutdown は新しいメッセージの受信を停止して処理中の応答のみ行ってからプロセスを終了します。（処理中のメッセージがなければ直ちに終了します）

### バックエンドアプリを再起動
アプリの更新などで再起動するには、まず `docker run ...` で新しいプロセスを起動します。
二重起動しても Slack のメッセージイベントは片方のプロセスに振り分けられるため正しく処理されます。

新しいプロセスが起動してメッセージの待ち受けを開始したら古いプロセスに TERM シグナルを送信して Graceful Shutdown することで停止時間なく再起動できます。
ただし、OssansNavi は1プロセスでの動作を前提とする最適化機能も含むため、ベストな応答のためには可能な限り1プロセスで運用してください。

## OssansNavi を開発する
### 開発環境
標準の開発環境は VSCode の DevContainers です。`.devcontainer/devcontainer.default.json` を `.devcontainer/devcontainer.json` にコピーしてご利用ください。

以下の解説は標準の開発環境を前提とします。

### 環境変数
開発環境では「3. バックエンドアプリの設定」と同様の環境変数を `.devcontainer/devcontainer.env` に設定します。

DevContainers で開発する場合は、ワークスペースをバインドマウントする際にホスト側の UID/GID とコンテナ内の UID/GID を合わせるために、以下のコマンドで `.devcontainer/.env` を作成します。

```bash
echo -e "UID=$(id -u)\nGID=$(id -g)" > .devcontainer/.env
```

OssansNavi を利用中の Slack ワークスペースで開発を行う場合は、開発用に別の OssansNavi Slack アプリを追加した上で以下の設定をします。
OSN_DEVELOPMENT_CHANNELS で指定したチャネルには本番モードのアプリが応答せず、開発モードのアプリのみ応答します。
開発モードのアプリは OSN_DEVELOPERS で指定したユーザーの DM にのみ応答します。

```properties
# -- 開発者向け設定 --
# Slack ワークスペース内で OssansNavi を利用しながら開発も行う場合に利用する設定
# OssansNavi を --production 引数なしで起動すると OSN_DEVELOPERS（カンマ区切りで複数可）に指定したユーザー以外の DM には応答しない
OSN_DEVELOPERS=UXXXXXXXX
# OssansNavi を --production 引数なしで起動すると OSN_DEVELOPMENT_CHANNELS（カンマ区切りで複数可）に指定したチャネルでのみ応答する
# OssansNavi を --production 引数ありで起動すると OSN_DEVELOPMENT_CHANNELS（カンマ区切りで複数可）に指定したチャネルでは応答しない
OSN_DEVELOPMENT_CHANNELS=GXXXXXXXX,CXXXXXXXXXX
```

## 最後に
OssansNavi は予告なく仕様変更する場合があります、ご了承ください。
