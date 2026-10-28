"""Request-local batching of overlapping graph reads across retrieval threads."""
from concurrent.futures import Future
from threading import Lock


class LiteratureReadCache:
    def __init__(self):
        self._lock = Lock()
        self._entries = {}

    def read_many(self, namespace, keys, fetch, row_key):
        """Reserve missing keys, fetch them together, and share in-flight results."""
        keys = list(dict.fromkeys(keys))
        pending, missing = {}, []
        with self._lock:
            for key in keys:
                identity = (namespace, key)
                future = self._entries.get(identity)
                if future is None:
                    future = self._entries[identity] = Future()
                    missing.append(key)
                pending[key] = future
        if missing:
            try:
                grouped = {key: [] for key in missing}
                for row in fetch(missing):
                    grouped[row[row_key]].append(row)
            except BaseException as error:
                for key in missing:
                    pending[key].set_exception(error)
                raise
            else:
                for key in missing:
                    pending[key].set_result(grouped[key])
        return [row for key in keys for row in pending[key].result()]

    def contains(self, namespace, key):
        """True when key is stored or being computed (compute_once will wait for it)."""
        with self._lock:
            return (namespace, key) in self._entries

    def compute_once(self, namespace, key, build):
        """Share an immutable processed value, including work still in flight."""
        identity = (namespace, key)
        with self._lock:
            future = self._entries.get(identity)
            owner = future is None
            if owner:
                future = self._entries[identity] = Future()
        if owner:
            try:
                value = build()
            except BaseException as error:
                future.set_exception(error)
                raise
            else:
                future.set_result(value)
        return future.result()
