import hashlib
from functools import lru_cache

class CacheManager:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.cache = {}
    
    def _hash_prompt(self, prompt, provider):
        return hashlib.md5(f"{provider}_{prompt}".encode()).hexdigest()
    
    def get_cached_response(self, prompt_hash):
        return self.cache.get(prompt_hash, None)

    def set_cached_response(self, prompt_hash, response):
        if len(self.cache) >= self.max_size:
            self.cache.clear()
        self.cache[prompt_hash] = response