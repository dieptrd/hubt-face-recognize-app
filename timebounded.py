from collections import deque, OrderedDict
import time
class TimeBounded:
    "Cache that invalidates and refreshes old entries."

    def __init__(self, maxage=60):
        self.cache = OrderedDict()      # { args : (timestamp, result)}
        self.maxage = maxage

    def add(self, key, value):
        self.cache[key] = time.time(), value
        self.cache.move_to_end(key)
        self._clear()

    def get(self, key):
        self._clear()
        if key in self.cache:
            _, result = self.cache[key]
            return result
        return None
    def exists(self, key):
        if key in self.cache:
            timestamp, _ = self.cache[key]
            if time.time() - timestamp <= self.maxage:
                return 1
        return 0
    
    def _first(self):
        key = next(iter(self.cache))
        timestamp, value = self.cache[key]
        return key, value, timestamp
    
    def _clear(self):
        while True:
            if len(self.cache) <= 0:
                break
            _, _, timestamp, = self._first()
            if time.time() - timestamp > self.maxage:
                self.cache.popitem(False)
            else:
                break 
    def count(self):
        self._clear()
        return len(self.cache)