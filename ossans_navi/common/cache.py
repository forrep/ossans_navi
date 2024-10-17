import time
from collections import OrderedDict
from typing import Generic, TypeVar


K = TypeVar('K')
V = TypeVar('V')


class LRUCache(Generic[K, V]):
    class NotFound:
        pass
    _NOT_FOUND = NotFound()

    def __init__(self, *, capacity: int, expire: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.expire = expire

    @staticmethod
    def is_found(value: V | NotFound):
        return value != LRUCache._NOT_FOUND

    @staticmethod
    def is_not_found(value: V | NotFound):
        return value == LRUCache._NOT_FOUND

    def get(self, key: K) -> V:
        if key not in self.cache or self.cache[key]["time"] < time.time():
            return LRUCache._NOT_FOUND
        return self.cache[key]["data"]

    def put(self, key: K, value: V):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = {"data": value, "time": time.time() + self.expire}
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self, key: K):
        if key in self.cache:
            del self.cache[key]