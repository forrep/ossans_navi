import asyncio
import re

import pytest

from ossans_navi.common import async_utils


@pytest.mark.asyncio
async def test_re_sub_simple_uppercase():
    async def replacer(match: re.Match[str]) -> str:
        await asyncio.sleep(0)  # yield control
        return match.group(0).upper()

    result = await async_utils.re_sub(r"\w+", replacer, "hello world")
    assert result == "HELLO WORLD"


@pytest.mark.asyncio
async def test_re_sub_count_limit():
    calls = []

    async def replacer(match: re.Match[str]) -> str:
        calls.append(match.group(0))
        return match.group(0).upper()

    # count=1 should only replace the first match
    result = await async_utils.re_sub(r"\w+", replacer, "one two three", count=1)
    assert result == "ONE two three"
    assert calls == ["one"]


@pytest.mark.asyncio
async def test_re_sub_concurrency_limit():
    order = []

    async def replacer(match: re.Match[str]) -> str:
        # record start
        order.append((match.group(0), "start"))
        # staggered sleep so that concurrency limits matter
        if match.group(0) == "a":
            await asyncio.sleep(0.08)
        elif match.group(0) == "b":
            await asyncio.sleep(0.04)
        order.append((match.group(0), "end"))
        return match.group(0).upper()

    # three matches: a, b, c
    text = "a b c"

    # concurrency=1 should run replacers sequentially
    result_seq = await async_utils.re_sub(r"\w+", replacer, text, concurrency=1)
    assert result_seq == "A B C"
    assert order == [("a", "start"), ("a", "end"), ("b", "start"), ("b", "end"), ("c", "start"), ("c", "end")]

    # reset order and test concurrency=3 (parallel)
    order.clear()
    result_par = await async_utils.re_sub(r"\w+", replacer, text, concurrency=3)
    assert result_par == "A B C"
    # with high concurrency, multiple 'start' entries should exist before 'end'
    assert order == [("a", "start"), ("b", "start"), ("c", "start"), ("c", "end"), ("b", "end"), ("a", "end")]


@pytest.mark.asyncio
async def test_asyncio_gather_concurrency():
    results = []

    async def worker(i: int) -> int:
        await asyncio.sleep(0.01 * (3 - i))
        results.append(i)
        return i * 2

    # concurrency=1 should run sequentially
    out_seq = await async_utils.asyncio_gather(worker(1), worker(2), worker(3), concurrency=1)
    assert out_seq == [2, 4, 6]
    assert results == [1, 2, 3]

    # concurrency=3 should run in parallel (order of results still preserved by gather)
    results.clear()
    out_par = await async_utils.asyncio_gather(worker(1), worker(2), worker(3), concurrency=3)
    assert out_par == [2, 4, 6]
    assert results == [3, 2, 1]
