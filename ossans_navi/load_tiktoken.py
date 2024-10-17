import tiktoken

# 各AIモデル用のトークン辞書をダウンロードする
# コンテナイメージ内に辞書を含める目的、つまり稼働環境で動的ダウンロードを抑止したい
if __name__ == "__main__":
    tiktoken.encoding_for_model("gpt-35-turbo")
    tiktoken.encoding_for_model("gpt-4")
    tiktoken.encoding_for_model("gpt-4o")
