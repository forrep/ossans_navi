FROM python:3.12.7-slim-bullseye AS base

RUN rm /etc/localtime \
    && echo "Asia/Tokyo" > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get update \
    && apt-get install -y locales curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i -e 's/# \(ja_JP.UTF-8\)/\1/' /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=ja_JP.UTF-8
ENV LANG="ja_JP.UTF-8"

ARG UID=20000
ARG GID=20000
ENV DOCKER_BUILD_UID=$UID DOCKER_BUILD_GID=$GID
RUN groupadd -g ${DOCKER_BUILD_GID} ossans_navi \
    && useradd -m -u ${DOCKER_BUILD_UID} -g ${DOCKER_BUILD_GID} -s /bin/bash -d /home/ossans_navi ossans_navi

USER ossans_navi
WORKDIR /home/ossans_navi/app

# Poetry がインストールされるディレクトリを事前に PATH に追加
ENV PATH=/home/ossans_navi/.local/bin:$PATH

# tiktoken が利用するトークン辞書ファイルのキャッシュ配置場所を環境変数で指定、事前DLするため
ENV TIKTOKEN_CACHE_DIR=/tmp/__tiktoken_cache__

# 変更頻度が低いファイルだけ先にコピーする（Dockerのキャッシュ効率UP）
COPY --chown=ossans_navi:ossans_navi pyproject.toml .
COPY --chown=ossans_navi:ossans_navi poetry.lock .

# Poetry のインストールと依存ライブラリのインストール ※時間がかかる
# ソースコードのコピー前に一度実行することでDockerのキャッシュがソースコード変更の影響を受けない
RUN curl -sSL https://install.python-poetry.org |python - \
    # bash を開いたタイミングで Poetry 仮想環境を有効化する処理。※プロジェクトフォルダに移動しないと poetry env info を実行できない
    && ( \
        echo 'if [ -f "$HOME/app/pyproject.toml" ] ; then' \
        && echo '    . $(cd $HOME/app; poetry env info --path)/bin/activate' \
        && echo 'fi' \
    ) |tee -a ~/.profile >> ~/.bashrc \
    # --no-root で root プロジェクトのインストールを行わない、なぜならば README やらアプリ本体などをまだ COPY していないからエラーになる
    && poetry install --no-root --only main

COPY --chown=ossans_navi:ossans_navi README.md .
COPY --chown=ossans_navi:ossans_navi ossans_navi ossans_navi

# ソースコード一式コピー後に ossans_navi の実行モジュールを仮想環境内にインストールするため再実行
# その前に実行した poetry install は pyproject.toml/poetry.lock だけで、かつ --no-root を付与しているため依存ライブラリのインストールのみで ossans_navi 自体のインストールが行われない
# 2段階に分ける理由は、ソースコード変更によって無効化される Docker イメージレイヤに、依存ライブラリをインストールするレイヤを含めたくないため（※ソースコードを変更しても依存ライブラリのレイヤをそのまま使いたい）
RUN poetry install --only main

# tiktoken のトークン辞書ファイルのキャッシュを事前DLする
RUN poetry run python ossans_navi/load_tiktoken.py

FROM base AS production
CMD ["poetry", "run", "python", "ossans_navi/app.py", "--production"]

FROM base AS development
USER root

RUN apt-get update \
    && apt-get install -y \
        sudo \
        less \
        git \
    && echo "ossans_navi ALL=(ALL:ALL) NOPASSWD:ALL" | EDITOR='tee -a' visudo

USER ossans_navi
# __pycache__ の生成ディレクトリを /tmp/__pycache__ とする
ENV PYTHONPYCACHEPREFIX=/tmp/__pycache__
# pytest のカバレッジデータファイルの生成ディレクトリを /tmp/ossan_coverage とする
ENV COVERAGE_DATA_FILE=/tmp/ossan_coverage/.coverage
# 開発用のライブラリをインストール
RUN poetry install
