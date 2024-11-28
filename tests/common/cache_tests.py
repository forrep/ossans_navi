import time

import pytest

from ossans_navi.common.cache import LRUCache


def test_lru_cache():
    cache = LRUCache[str, str](capacity=3, expire=2)
    cache.put('a', "あ")
    cache.put('b', "い")
    cache.put('c', "う")
    assert (v := cache.get("a")).found
    assert v.value == "あ"

    assert (v := cache.get("b")).found
    assert v.value == "い"

    assert (v := cache.get("c")).found
    assert v.value == "う"

    assert not (v := cache.get("d")).found
    with pytest.raises(ValueError):
        v.value

    cache.put('d', "え")

    assert not (v := cache.get("a")).found
    with pytest.raises(ValueError):
        v.value

    assert (v := cache.get("b")).found
    assert v.value == "い"

    assert (v := cache.get("c")).found
    assert v.value == "う"

    assert (v := cache.get("d")).found
    assert v.value == "え"

    # 2.5秒待って有効期限2秒のキャッシュを無効化させる
    time.sleep(2.5)

    assert not cache.get("a").found
    assert not cache.get("b").found
    assert not cache.get("c").found
    assert not cache.get("d").found
