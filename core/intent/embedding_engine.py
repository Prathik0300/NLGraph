"""
core/intent/embedding_engine.py — bge-m3 embedding engine via FlagEmbedding.

Produces THREE vector types from a single forward pass:
  - dense  (1024-dim)  : semantic similarity
  - sparse (token→weight): lexical / BM25-style matching
  - colbert (token-level): fine-grained reranking (optional)

Cache: in-memory LRU with TTL.  Key = SHA256(normalised text).
No Ollama dependency for embeddings.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import settings
from core.exceptions import EmbeddingError, ModelNotCachedError
from core.models import HybridVector


# ══════════════════════════════════════════════════════════════════════════
# EMBED CACHE
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class CacheEntry:
    dense:      np.ndarray
    sparse:     dict[str, float]
    inserted_at: float = field(default_factory=time.time)


class EmbedCache:
    """
    In-memory LRU cache with TTL for embedding results.
    Key: SHA256(text.strip().lower())
    Max size enforced by evicting least-recently-used entries.
    """

    def __init__(self, max_size: int, ttl_seconds: int):
        self._max_size   = max_size
        self._ttl        = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits        = 0
        self._misses      = 0
        self._evictions   = 0
        self._lock        = asyncio.Lock()

    @staticmethod
    def _key(text: str):
        normalised = text.strip().lower()
        return hashlib.sha256(normalised.encode()).hexdigest()

    async def get(self, text: str):
        key = self._key(text)
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            # TTL check
            if time.time() - entry.inserted_at > self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            # LRU: move to end
            self._store.move_to_end(key)
            self._hits += 1
            return entry

    async def put(self, text: str, dense: np.ndarray, sparse: dict[str, float]):
        key = self._key(text)
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = CacheEntry(dense=dense, sparse=sparse)
            # Evict oldest if over capacity
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)
                self._evictions += 1

    async def invalidate_all(self):
        async with self._lock:
            self._store.clear()

    def stats(self):
        total = self._hits + self._misses
        return {
            "hits":       self._hits,
            "misses":     self._misses,
            "hit_rate":   self._hits / total if total else 0.0,
            "size":       len(self._store),
            "evictions":  self._evictions,
        }


# ══════════════════════════════════════════════════════════════════════════
# EMBEDDING ENGINE
# ══════════════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """
    Wraps BAAI/bge-m3 via FlagEmbedding.
    Thread-safe via asyncio.  All heavy work runs in an executor
    (releases the GIL — bge-m3 is PyTorch C++ under the hood).

    Usage:
        engine = EmbeddingEngine()
        await engine.initialize()
        vec = await engine.embed("research transformer architectures")
        hybrid = await engine.embed_hybrid("research transformer architectures")
    """

    def __init__(self):
        self._model    = None   # BGEM3FlagModel, loaded lazily
        self._cache    = EmbedCache(
            max_size    = settings.EMBED_CACHE_MAX_SIZE,
            ttl_seconds = settings.EMBED_CACHE_TTL_SECONDS,
        )
        self._loop     = None
        self._executor = None   # ThreadPoolExecutor for blocking model calls
        self._init_lock = asyncio.Lock()
        self._ready    = False

    # ── Initialisation ────────────────────────────────────────────────────

    async def initialize(self):
        """Load the model.  Call once at startup."""
        async with self._init_lock:
            if self._ready:
                return
            await asyncio.get_event_loop().run_in_executor(
                None, self._load_model
            )
            self._ready = True

    def _load_model(self):
        """Blocking: loads bge-m3.  Runs in executor thread."""
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise EmbeddingError(
                "FlagEmbedding not installed. Run: pip install FlagEmbedding"
            ) from e

        try:
            self._model = BGEM3FlagModel(
                settings.EMBED_MODEL,
                use_fp16=True,              # half-precision on CPU: ~2x speedup
                cache_dir=str(settings.MODEL_CACHE_DIR),
                local_files_only=settings.LOCAL_FILES_ONLY,
            )
        except OSError as e:
            if settings.LOCAL_FILES_ONLY:
                raise ModelNotCachedError(
                    f"Model '{settings.EMBED_MODEL}' not found in cache "
                    f"({settings.MODEL_CACHE_DIR}). Set LOCAL_FILES_ONLY=False "
                    f"to download it."
                ) from e
            raise EmbeddingError(f"Failed to load model: {e}") from e

    async def _ensure_ready(self):
        if not self._ready:
            await self.initialize()

    # ── Core embed operations ─────────────────────────────────────────────

    async def embed(self, text: str):
        """
        Embed a single text → 1024-dim L2-normalised dense vector.
        Cached.
        """
        await self._ensure_ready()

        # Cache check
        cached = await self._cache.get(text)
        if cached is not None:
            return cached.dense

        # Run embedding in executor (releases GIL)
        dense, sparse = await asyncio.get_event_loop().run_in_executor(
            None, self._encode_one, text
        )

        await self._cache.put(text, dense, sparse)
        return dense

    async def embed_batch(self, texts: list[str]):
        """
        Embed a list of texts → shape (N, 1024) dense matrix.
        Pulls cached items, batches uncached ones.
        """
        await self._ensure_ready()

        results   = [None] * len(texts)
        uncached_indices: list[int]  = []
        uncached_texts:   list[str]  = []

        # Cache lookup pass
        for i, text in enumerate(texts):
            cached = await self._cache.get(text)
            if cached is not None:
                results[i] = cached.dense
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch encode uncached
        if uncached_texts:
            batch_size = settings.EMBED_BATCH_SIZE
            for chunk_start in range(0, len(uncached_texts), batch_size):
                chunk = uncached_texts[chunk_start: chunk_start + batch_size]
                chunk_idx = uncached_indices[chunk_start: chunk_start + batch_size]

                dense_batch, sparse_batch = await asyncio.get_event_loop().run_in_executor(
                    None, self._encode_batch, chunk
                )

                for j, (orig_idx, text) in enumerate(zip(chunk_idx, chunk)):
                    d = dense_batch[j]
                    s = sparse_batch[j] if sparse_batch else {}
                    await self._cache.put(text, d, s)
                    results[orig_idx] = d

        return np.stack(results)

    async def embed_hybrid(self, text: str):
        """
        Embed text → HybridVector with both dense and sparse components.
        Used for Qdrant hybrid search.
        """
        await self._ensure_ready()

        cached = await self._cache.get(text)
        if cached is not None:
            return HybridVector(
                dense=cached.dense.tolist(),
                sparse=cached.sparse,
            )

        dense, sparse = await asyncio.get_event_loop().run_in_executor(
            None, self._encode_one, text
        )
        await self._cache.put(text, dense, sparse)
        return HybridVector(dense=dense.tolist(), sparse=sparse)

    # ── Low-level encoding (blocking, run in executor) ────────────────────

    def _encode_one(self, text: str):
        """Encode a single text. Blocking."""
        output = self._model.encode(
            [text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=1,
        )
        dense  = self._l2_normalise(output["dense_vecs"][0])
        sparse = self._extract_sparse(output["lexical_weights"][0])
        return dense, sparse

    def _encode_batch(
        self, texts: list[str]
    ):
        """Encode a batch. Blocking."""
        output = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=len(texts),
        )
        dense_batch = np.array([
            self._l2_normalise(v) for v in output["dense_vecs"]
        ])
        sparse_batch = [
            self._extract_sparse(lw)
            for lw in output["lexical_weights"]
        ]
        return dense_batch, sparse_batch

    @staticmethod
    def _l2_normalise(vec: np.ndarray):
        """L2-normalise a vector to unit length."""
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec
        return vec / norm

    @staticmethod
    def _extract_sparse(lexical_weights: dict):
        """
        Convert bge-m3 lexical_weights output to a plain {token_id_str: weight} dict.
        Zero-weight entries are dropped.
        """
        result = {}
        for token_id, weight in lexical_weights.items():
            if weight > 0.0:
                result[str(token_id)] = float(weight)
        return result

    # ── Health check ─────────────────────────────────────────────────────

    async def health_check(self):
        """Returns True if the model can produce embeddings."""
        try:
            await self.initialize()
            test = await self.embed("health check")
            return test.shape == (settings.EMBED_DIM,)
        except Exception:
            return False

    # ── Cache access ──────────────────────────────────────────────────────

    def cache_stats(self):
        return self._cache.stats()

    async def clear_cache(self):
        await self._cache.invalidate_all()


# ── Module-level singleton ────────────────────────────────────────────────
# Imported by all pipeline stages: from core.intent.embedding_engine import embed_engine
embed_engine = EmbeddingEngine()


if __name__ == "__main__":
    """Quick smoke test — run: python -m core.intent.embedding_engine"""
    import asyncio

    async def demo():
        print("Loading bge-m3…")
        await embed_engine.initialize()
        print("Model loaded.")

        texts = [
            "research transformer architectures",
            "understand the attention mechanism deeply",
            "implement a simple version in Python",
            "test the implementation properly",
        ]

        print(f"\nEmbedding {len(texts)} phrases…")
        matrix = await embed_engine.embed_batch(texts)
        print(f"Dense matrix shape: {matrix.shape}")

        # Cosine similarity matrix (inner product of normalised vectors)
        sim = matrix @ matrix.T
        print("\nCosine similarity matrix:")
        header = "".join(f"{i:8}" for i in range(len(texts)))
        print(f"   {header}")
        for i, row in enumerate(sim):
            vals = "".join(f"{v:8.3f}" for v in row)
            print(f"{i}  {vals}")

        # Hybrid
        print("\nHybrid vector for 'implement a sorting algorithm'…")
        hv = await embed_engine.embed_hybrid("implement a sorting algorithm")
        print(f"  Dense dim:   {len(hv.dense)}")
        print(f"  Sparse non-zero tokens: {len(hv.sparse)}")
        top5 = sorted(hv.sparse.items(), key=lambda x: -x[1])[:5]
        print(f"  Top-5 sparse tokens: {top5}")

        print("\nCache stats:", embed_engine.cache_stats())
        print("\nHealth check:", await embed_engine.health_check())

    asyncio.run(demo())
