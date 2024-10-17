import time

from ossans_navi.common.cache import LRUCache


def test_lru_cache():
    cache = LRUCache[str, str](capacity=3, expire=2)
    cache.put('a', "あ")
    cache.put('b', "い")
    cache.put('c', "う")
    assert LRUCache.is_found(cache.get("a"))
    assert not LRUCache.is_not_found(cache.get("a"))
    assert cache.get("a") == "あ"

    assert LRUCache.is_found(cache.get("b"))
    assert not LRUCache.is_not_found(cache.get("b"))
    assert cache.get("b") == "い"

    assert LRUCache.is_found(cache.get("c"))
    assert not LRUCache.is_not_found(cache.get("c"))
    assert cache.get("c") == "う"

    cache.put('d', "え")

    assert not LRUCache.is_found(cache.get("a"))
    assert LRUCache.is_not_found(cache.get("a"))
    assert cache.get("a") != "あ"

    assert LRUCache.is_found(cache.get("b"))
    assert not LRUCache.is_not_found(cache.get("b"))
    assert cache.get("b") == "い"

    assert LRUCache.is_found(cache.get("c"))
    assert not LRUCache.is_not_found(cache.get("c"))
    assert cache.get("c") == "う"

    assert LRUCache.is_found(cache.get("d"))
    assert not LRUCache.is_not_found(cache.get("d"))
    assert cache.get("d") == "え"

    time.sleep(2.1)

    assert LRUCache.is_not_found(cache.get("a"))
    assert LRUCache.is_not_found(cache.get("b"))
    assert LRUCache.is_not_found(cache.get("c"))
    assert LRUCache.is_not_found(cache.get("d"))
