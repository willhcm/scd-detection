from collections import deque
from concurrent.futures import ThreadPoolExecutor

def prefetch(fn, items, depth=1):
    items = iter(items)
    q = deque()
    with ThreadPoolExecutor(1) as ex:
        for it in items:
            q.append((it, ex.submit(fn, it)))
            if len(q) > depth:
                break
        while q:
            it, f = q.popleft()
            nxt = next(items, None)
            if nxt is not None:
                q.append((nxt, ex.submit(fn, nxt)))
            yield it, f.result()