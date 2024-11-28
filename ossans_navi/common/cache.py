import dataclasses
import time
from collections import OrderedDict
from typing import Generic, Optional, TypeVar

K = TypeVar('K')
V = TypeVar('V')


@dataclasses.dataclass
class CacheEntry(Generic[V]):
    _value: Optional[V]
    _expired: float
    found: bool

    @property
    def value(self) -> V:
        if not self.found:
            raise ValueError('Not found Error')
        return self._value  # type: ignore


class LRUCache(Generic[K, V]):
    class NotFound:
        pass
    _NOT_FOUND = NotFound()

    def __init__(self, *, capacity: int, expire: int):
        self.cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self.capacity = capacity
        self.expire = expire
        self.null: CacheEntry[V] = CacheEntry(None, 0, False)

    def get(self, key: K) -> CacheEntry[V]:
        if key not in self.cache or self.cache[key]._expired < time.time():
            return self.null
        return self.cache[key]

    def put(self, key: K, value: V):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = CacheEntry(value, time.time() + self.expire, True)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self, key: K):
        if key in self.cache:
            del self.cache[key]
