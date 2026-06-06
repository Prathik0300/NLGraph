"""
core/memory/global_store.py — Qdrant-backed persistent global memory.

3 collections:
  intent_vectors  : every processed query + plan + feedback score
  agent_outputs   : high-quality agent outputs (score >= 4.0) by capability
  quality_vectors : user corrections (positive + negative examples)
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime

import numpy as np

from config import settings
from core.models import IntentMap, HybridVector


class GlobalStore:

    def __init__(self):
        self._client  = None
        self._ready   = False
        self._lock    = asyncio.Lock()

    async def initialize(self):
        async with self._lock:
            if self._ready:
                return
            await asyncio.get_event_loop().run_in_executor(None, self._init_sync)
            self._ready = True

    def _init_sync(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams, SparseVectorParams, Modifier,
        )
        settings.ensure_dirs()
        self._client = QdrantClient(path=str(settings.QDRANT_PATH))

        # Create collections if they don't exist
        existing = {c.name for c in self._client.get_collections().collections}

        vector_config = {
            "dense":  VectorParams(size=settings.EMBED_DIM, distance=Distance.COSINE),
        }
        sparse_config = {
            "sparse": SparseVectorParams(modifier=Modifier.IDF),
        }

        for name in ["intent_vectors", "agent_outputs", "quality_vectors"]:
            if name not in existing:
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=vector_config,
                    sparse_vectors_config=sparse_config,
                )

    # ── Intent store ──────────────────────────────────────────────────────

    async def store_intent(self, query: str, hybrid: HybridVector, plan: IntentMap, feedback_score: float = None):
        if not self._ready:
            return
        plan_json = plan.model_dump_json()
        # Cap payload at 16 KB
        if len(plan_json) > 16_384:
            plan_json = json.dumps({"truncated": True, "query_id": plan.query_id})

        point_id = abs(hash(query + str(time.time()))) % (2**31)
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._upsert_intent(point_id, hybrid, plan_json, query, feedback_score)
        )

    def _upsert_intent(self, point_id, hybrid, plan_json, query, feedback_score):
        from qdrant_client.models import PointStruct, SparseVector
        self._client.upsert(
            collection_name="intent_vectors",
            points=[PointStruct(
                id=point_id,
                vector={
                    "dense":  hybrid.dense,
                    "sparse": SparseVector(
                        indices=[int(k) for k in hybrid.sparse.keys()],
                        values=list(hybrid.sparse.values()),
                    ),
                },
                payload={
                    "query":          query,
                    "plan_json":      plan_json,
                    "feedback_score": feedback_score,
                    "timestamp":      time.time(),
                },
            )]
        )

    async def lookup_intent(self, hybrid: HybridVector):
        """
        Hybrid search for similar past queries.
        Returns (similarity, cached_plan_json) or None.
        """
        if not self._ready:
            return None
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._search_intent(hybrid)
            )
            if result and result[0].score >= settings.CACHE_HIT_THRESHOLD:
                return result[0].score, result[0].payload.get("plan_json"), result[0].payload.get("feedback_score")
        except Exception:
            pass
        return None

    def _search_intent(self, hybrid):
        # Cache-hit detection must use dense cosine similarity — NOT RRF fusion.
        # RRF scores are rank-based and Qdrant normalises the top result to ~1.0
        # regardless of actual semantic equivalence, so any topically related
        # query would trigger a false cache hit against a 0.91-style threshold.
        # Dense cosine similarity is a true [0, 1] metric: 0.97+ means the
        # queries are essentially the same question.
        return self._client.search(
            collection_name="intent_vectors",
            query_vector=("dense", hybrid.dense),
            limit=3,
            with_payload=True,
        )

    # ── Agent output store ────────────────────────────────────────────────

    async def store_agent_output(self, capability: str, domain: str, content: str,
                                  quality_score: float, hybrid: HybridVector, query_id: str):
        if not self._ready or quality_score < settings.AGENT_OUTPUT_MIN_QUALITY:
            return
        point_id = abs(hash(content[:100] + query_id)) % (2**31)
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._upsert_output(point_id, hybrid, capability, domain, content, quality_score, query_id)
        )

    def _upsert_output(self, point_id, hybrid, capability, domain, content, quality_score, query_id):
        from qdrant_client.models import PointStruct, SparseVector
        self._client.upsert(
            collection_name="agent_outputs",
            points=[PointStruct(
                id=point_id,
                vector={
                    "dense":  hybrid.dense,
                    "sparse": SparseVector(
                        indices=[int(k) for k in hybrid.sparse.keys()],
                        values=list(hybrid.sparse.values()),
                    ),
                },
                payload={
                    "capability":    capability,
                    "domain":        domain,
                    "content":       content[:2000],
                    "quality_score": quality_score,
                    "query_id":      query_id,
                    "timestamp":     time.time(),
                },
            )]
        )

    async def get_relevant_outputs(self, capability: str, query_vector: list, top_k: int = 3):
        if not self._ready:
            return []
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.search(
                    collection_name="agent_outputs",
                    query_vector=("dense", query_vector),
                    query_filter={
                        "must": [
                            {"key": "capability", "match": {"value": capability}},
                            {"key": "quality_score", "range": {"gte": settings.AGENT_OUTPUT_MIN_QUALITY}},
                        ]
                    },
                    limit=top_k,
                    with_payload=True,
                )
            )
            return [r.payload.get("content", "") for r in results]
        except Exception:
            return []

    # ── Stats ─────────────────────────────────────────────────────────────

    async def count_intents(self):
        if not self._ready:
            return 0
        try:
            info = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._client.get_collection("intent_vectors")
            )
            return info.points_count
        except Exception:
            return 0


# ── Singleton ─────────────────────────────────────────────────────────────
_global_store = None

def get_global_store():
    global _global_store
    if _global_store is None:
        _global_store = GlobalStore()
    return _global_store
