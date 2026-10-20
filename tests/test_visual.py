import numpy as np

from grstbx.visual import _LRUCache


def test_lru_cache_bounded():
    cache = _LRUCache(max_bytes=3 * 800)
    for key in range(5):
        cache.put(key, np.zeros(100))             # 800 bytes each
    assert list(cache) == [2, 3, 4]
    assert cache.nbytes == 2400
    cache.get(2)                                   # 2 becomes the most recent
    cache.put(5, (np.zeros(100), np.zeros(0)))
    assert list(cache) == [4, 2, 5]
