import time, json
from config import REDIS_URL, CACHE_TTL

try:
    import redis
except Exception:
    redis = None

class InMemoryCache:
    def __init__(self):
        self.store = {}
    def get(self, key):
        v = self.store.get(key)
        if not v:
            return None
        val, exp = v
        if time.time() > exp:
            del self.store[key]
            return None
        return val
    def set(self, key, value, ttl=CACHE_TTL):
        self.store[key] = (value, time.time()+ttl)

class RedisCache:
    def __init__(self, url):
        self.client = redis.from_url(url)
    def get(self, key):
        v = self.client.get(key)
        return v.decode('utf-8') if v else None
    def set(self, key, value, ttl=CACHE_TTL):
        self.client.set(key, value, ex=ttl)

def make_cache():
    if REDIS_URL and redis:
        return RedisCache(REDIS_URL)
    return InMemoryCache()
