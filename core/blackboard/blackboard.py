"""
core/blackboard/blackboard.py — Per-query shared knowledge space.

All agents for a query read and write here.
Facts are append-only and indexed in FAISS for semantic retrieval.
Concurrency: single asyncio.Lock for mutations (all agents are async coroutines).
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np

from config import settings
from core.models import (
    Agent, AgentState, AgentStatus, Artifact,
    BlackboardSnapshot, Decision, Fact, IntentMap, LogEntry,
)


class Blackboard:

    def __init__(self, query_id: str, original_query: str, intent_map: IntentMap):
        self.query_id       = query_id
        self.original_query = original_query
        self.intent_map     = intent_map

        # Append-only public state
        self.facts:          list[Fact]              = []
        self.decisions:      list[Decision]          = []
        self.artifacts:      list[Artifact]          = []
        self.open_questions: list[str]               = []
        self.agent_states:   dict[str, AgentState]   = {}
        self.execution_log:  list[LogEntry]          = []

        # Internal: FAISS index + mapping
        self._faiss_index   = None
        self._fact_id_map:  list[str]  = []   # faiss position → fact_id
        self._facts_by_id:  dict[str, Fact] = {}
        self._lock          = asyncio.Lock()

        # Initialise FAISS
        self._init_faiss()

        # Initialise agent states to PENDING
        for agent in intent_map.agents:
            self.agent_states[agent.agent_id] = AgentState(
                agent_id=agent.agent_id,
                status=AgentStatus.PENDING,
            )

    def _init_faiss(self):
        try:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(settings.EMBED_DIM)
        except ImportError:
            self._faiss_index = None   # graceful fallback — semantic search disabled

    # ── Write operations (lock-protected) ────────────────────────────────

    async def add_fact(self, fact: Fact):
        async with self._lock:
            self.facts.append(fact)
            self._facts_by_id[fact.fact_id] = fact

            if fact.embedding and self._faiss_index is not None:
                vec = np.array(fact.embedding, dtype=np.float32).reshape(1, -1)
                self._faiss_index.add(vec)
                self._fact_id_map.append(fact.fact_id)

            # Evict oldest 20% if over capacity
            if len(self.facts) > settings.FAISS_MAX_FACTS:
                await self._evict_oldest()

    async def add_decision(self, decision: Decision):
        async with self._lock:
            self.decisions.append(decision)

    async def add_artifact(self, artifact: Artifact):
        async with self._lock:
            self.artifacts.append(artifact)

    async def add_open_question(self, question: str):
        async with self._lock:
            if question not in self.open_questions:
                self.open_questions.append(question)

    async def update_agent_state(self, agent_id: str, state: AgentState):
        async with self._lock:
            self.agent_states[agent_id] = state

    async def log(self, event: str, agent_id: str = None, level: str = "info", **details):
        async with self._lock:
            self.execution_log.append(LogEntry(
                event=event,
                agent_id=agent_id,
                level=level,
                details=details,
            ))

    # ── Read operations (no lock needed for reads) ────────────────────────

    async def search_facts(self, query_vector: np.ndarray, top_k: int = None):
        """Semantic search over blackboard facts using FAISS."""
        top_k = top_k or settings.CONTEXT_TOP_K_FACTS

        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return self.facts[:top_k]

        qv = query_vector.astype(np.float32).reshape(1, -1)
        k  = min(top_k, self._faiss_index.ntotal)
        distances, indices = self._faiss_index.search(qv, k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self._fact_id_map):
                fid  = self._fact_id_map[idx]
                fact = self._facts_by_id.get(fid)
                if fact:
                    results.append(fact)

        return results

    def get_upstream_artifacts(self, agent: Agent):
        """Return all artifacts produced by upstream agents."""
        dep_ids = set(agent.depends_on)
        return [a for a in self.artifacts
                if a.produced_by in dep_ids]

    def get_artifact_by_key(self, key: str):
        """Return the most recent artifact with a given key."""
        matches = [a for a in self.artifacts if a.artifact_key == key]
        return matches[-1] if matches else None

    def snapshot(self):
        """Return a serialisable copy (no FAISS, no locks)."""
        return BlackboardSnapshot(
            query_id=self.query_id,
            original_query=self.original_query,
            intent_map=self.intent_map,
            facts=list(self.facts),
            decisions=list(self.decisions),
            artifacts=list(self.artifacts),
            open_questions=list(self.open_questions),
            agent_states=dict(self.agent_states),
            execution_log=list(self.execution_log),
        )

    # ── Private ───────────────────────────────────────────────────────────

    async def _evict_oldest(self):
        """Evict oldest 20% of facts. Rebuilds FAISS index."""
        n_keep  = int(len(self.facts) * 0.80)
        self.facts          = self.facts[-n_keep:]
        self._fact_id_map   = [f.fact_id for f in self.facts if f.embedding]
        self._facts_by_id   = {f.fact_id: f for f in self.facts}

        if self._faiss_index is not None:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(settings.EMBED_DIM)
            vecs = [
                np.array(f.embedding, dtype=np.float32)
                for f in self.facts if f.embedding
            ]
            if vecs:
                matrix = np.stack(vecs)
                self._faiss_index.add(matrix)
