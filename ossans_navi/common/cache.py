import time
from collections import OrderedDict
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

K = TypeVar('K')
V = TypeVar('V')


class CacheEntry(BaseModel, Generic[V]):
    v: Optional[V]
    expired: float
    found: bool

    @property
    def value(self) -> V:
        if not self.found:
            raise ValueError('Not found Error')
        return self.v  # type: ignore


class LRUCache(Generic[K, V]):
    class NotFound:
        pass
    _NOT_FOUND = NotFound()

    def __init__(self, *, capacity: int, expire: int):
        self.cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self.capacity = capacity
        self.expire = expire
        self.null: CacheEntry[V] = CacheEntry(v=None, expired=0, found=False)

    def get(self, key: K) -> CacheEntry[V]:
        if key not in self.cache or self.cache[key].expired < time.time():
            return self.null
        return self.cache[key]

    def put(self, key: K, value: V):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = CacheEntry(v=value, expired=time.time() + self.expire, found=True)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self, key: K):
        if key in self.cache:
            del self.cache[key]
