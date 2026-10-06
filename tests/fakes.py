import hashlib
import math
import re

from threadlight.db.models import EMBEDDING_DIM


class HashingEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words have similar vectors."""

    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * EMBEDDING_DIM
        v[0] = 0.01  # never a zero vector (cosine distance would be undefined)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [self._vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
