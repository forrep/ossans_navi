# OssansNavi
OssansNavi は起動するだけで Slack ワークスペースに蓄積された情報を活用して応答するチャットボットになります。

必要なのは普段利用している Slack ワークスペースと OpenAI (Azure OpenAI) の API だけです。

## 主な機能
- Slack のパブリックチャネルを情報源として応答します
- プライベートチャネルはデフォルトでは参照しません（特定のプライベートチャネルを参照する設定あり）
- パブリックチャネルの参照に制限のあるユーザー（ゲスト等）には応答しません
- OssansNavi のボットユーザーが参加しているチャネルや DM で応答します
- OssansNavi をメンションすると確実に応答してくれます
- OssansNavi をメンションしなくても応答可能ならば応答します
  - まずは様々なチャネルに OssansNavi のボットユーザーを追加してみることをオススメします
- OssansNavi が質問に回答できない場合は詳しい人にメンションすることもあります。メンションされた方は広い心で対応しましょう
- 応答に 1~3分はかかるので相手を人間だと思って気長にお待ちください

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
任意のディレクトリを作成して、以下の `.env` ファイルを用意します。

OSN_SLACK_APP_TOKEN, OSN_SLACK_BOT_TOKEN, OSN_SLACK_USER_TOKEN には、前の手順で発行したトークンをそれぞれ設定します。
LLM には openai, azure_openai のどちらかを利用可能です。

```properties
# Slack トークン
OSN_SLACK_APP_TOKEN=xapp-...
OSN_SLACK_BOT_TOKEN=xoxb-...
OSN_SLACK_USER_TOKEN=xoxp-...

# 利用するLLMの種類（openai, azure_openai）
OSN_AI_SERVICE_TYPE=azure_openai

# -- Azure OpenAI の設定 --
# Azure OpenAI APIキー
OSN_AZURE_OPENAI_API_KEY=684...
# Azure OpenAI EndPoint ドメイン
OSN_AZURE_OPENAI_ENDPOINT=https://*.openai.azure.com/
# Azure OpenAI 低コストモデルの DEPLOYMENT_NAME
OSN_AZURE_OPENAI_MODEL_LOW_COST=gpt-4o-mini
# Azure OpenAI 低コストモデルの入力コスト（1,000,000 tokens あたり） ※任意の通貨単位を設定可 
OSN_AZURE_OPENAI_MODEL_LOW_COST_IN=0.15
# Azure OpenAI 低コストモデルの出力コスト（1,000,000 tokens あたり） ※任意の通貨単位を設定可
OSN_AZURE_OPENAI_MODEL_LOW_COST_OUT=0.60
# Azure OpenAI 高クオリティモデルの DEPLOYMENT_NAME
OSN_AZURE_OPENAI_MODEL_HIGH_QUALITY=gpt-4o
# Azure OpenAI 高クオリティモデルの入力コスト（1,000,000 tokens あたり） ※任意の通貨単位を設定可
OSN_AZURE_OPENAI_MODEL_HIGH_QUALITY_IN=2.50
# Azure OpenAI 高クオリティモデルの出力コスト（1,000,000 tokens あたり） ※任意の通貨単位を設定可
OSN_AZURE_OPENAI_MODEL_HIGH_QUALITY_OUT=10.0

# -- アプリの設定 --
# Slack ワークスペースの名称、典型的には社名等を指定、システムプロンプトで入力されて OssansNavi が自身の稼働する環境を認識するために利用
OSN_WORKSPACE_NAME=ABC株式会社
# OssansNavi の呼称、Slack ワークスペース内で OssansNavi がなんと呼ばれるかを指定（カンマ区切りで複数可）、OssansNavi が自身への呼びかけや文脈を理解するために利用する
OSN_ASSISTANT_NAMES=OssansNavi,おっさんずナビ,おっさん
# 言語、OssansNavi が Slack ワークスペース内の検索及び応答に利用する言語、省略時はユーザーの言語になるべく合わせて応答
OSN_LANGUAGE=Japanese
# ログレベル
OSN_LOG_LEVEL=WARN
# OssansNavi の応答をロギングするチャネル ※省略可 例: {"channel": "CXXXXXXXX", "thread_ts", "cost": 0.04}
# OSN_RESPONSE_LOGGING_CHANNEL=
```

OSN_AI_SERVICE_TYPE=openai とした場合は OSN_AZURE_ の代わりに、以下の OpenAI の設定をします。

```properties
OSN_AI_SERVICE_TYPE=openai
OSN_OPENAI_API_KEY=684...
OSN_OPENAI_MODEL_LOW_COST=gpt-4o-mini
OSN_OPENAI_MODEL_LOW_COST_IN=0.15
OSN_OPENAI_MODEL_LOW_COST_OUT=0.60
OSN_OPENAI_MODEL_HIGH_QUALITY=gpt-4o
OSN_OPENAI_MODEL_HIGH_QUALITY_IN=2.50
OSN_OPENAI_MODEL_HIGH_QUALITY_OUT=10.0
```

### 4. バックエンドアプリの起動
```
docker build --target production -t ossans_navi:latest .
docker run --env-file .env ossans_navi:latest
```

起動に成功すると INFO レベルのログで以下のメッセージが表示されます。（※ OSN_LOG_LEVEL=INFO 以上が必要）

```
slack_bolt(265) - INFO - Starting to receive messages from a new connection 
```

まずは Slack アプリで OssansNavi アプリを開いて DM で話しかけてみましょう。
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

また、開発環境ではいくつかの追加設定があります。

```properties
# DevContainers用の設定（開発時のみ必要）
# ワークスペースをバインドマウントする際に、ホスト側の UID/GID とコンテナ内の UID/GID を合わせるために必要
UID=1000
GID=1000
```

OssansNavi をすでに利用している Slack ワークスペースで開発も行う場合は、開発用に別の Slack アプリを追加した上で以下の設定をします。
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
