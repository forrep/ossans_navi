import asyncio
import re
from asyncio.futures import Future
from typing import Awaitable, Callable, Pattern, TypeVar

T = TypeVar('T')


async def re_sub(
    pattern: str | Pattern[str],
    repl: Callable[[re.Match[str]], Awaitable[str]],
    string: str,
    count: int = 0,
    flags: int = 0,
    concurrency: int = 1
) -> str:
    """
    re.sub の async 版。第二引数に async 関数を取れる。

    Args:
        pattern: 正規表現パターン（文字列または re.Pattern）
        repl: マッチオブジェクトを受け取り、置換文字列を返す async 関数
        string: 対象の文字列
        count: 置換する最大回数（0 = 無制限）
        flags: 正規表現フラグ（re.IGNORECASE など）
        concurrency: 並列実行数の上限（0 = 無制限）

    Returns:
        置換後の文字列

    Example:
        async def replacer(match: re.Match[str]) -> str:
            # 非同期処理（例: API呼び出し）
            await asyncio.sleep(0.1)
            return match.group(0).upper()

        result = await re_sub_async(r'\\w+', replacer, 'hello world')
        # result: 'HELLO WORLD'

        # 並列数を制限する場合
        result = await re_sub_async(r'\\w+', replacer, 'hello world', concurrency=5)
    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern, flags)

    # マッチをすべて見つける
    matches = list(pattern.finditer(string))

    # count が指定されている場合は制限する
    if count > 0:
        matches = matches[:count]

    if not matches:
        return string

    # すべてのマッチに対して非同期に置換文字列を取得
    if concurrency > 0:
        # 並列数を制限する場合
        semaphore = asyncio.Semaphore(concurrency)

        async def repl_wrapper(match: re.Match[str]) -> str:
            async with semaphore:
                return await repl(match)
    else:
        # 並列数無制限
        async def repl_wrapper(match: re.Match[str]) -> str:
            return await repl(match)

    replacements = await asyncio.gather(*[repl_wrapper(match) for match in matches])

    # 後ろから順に置換（インデックスがずれないように）
    result = string
    for match, replacement in reversed(list(zip(matches, replacements))):
        result = result[:match.start()] + replacement + result[match.end():]

    return result


async def gather_wrapper(coros_or_futures: Future[T] | Awaitable[T], semaphore: asyncio.Semaphore) -> T:
    async with semaphore:
        return await coros_or_futures


async def asyncio_gather(
    *coros_or_futures: Future[T] | Awaitable[T],
    concurrency: int = 1
) -> list[T]:
    """
    asyncio.Semaphore で同時実行数を制限しつつ asyncio.gather 相当のことをするヘルパー。
    """
    semaphore = asyncio.Semaphore(concurrency)

    return await asyncio.gather(
        *(gather_wrapper(c, semaphore) for c in coros_or_futures),
        return_exceptions=False,
    )
