def decode(content: bytes) -> str:
    """content を UTF-8/SJIS としてデコードする"""
    for encoding in ("utf-8", "cp932"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    # 必ずエラーとなる、UnicodeDecodeError を発生させる
    return content.decode("utf-8")
