"""
core/memory/performance_tracker.py — SQLite-backed performance + feedback tracking.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from config import settings
from core.models import GlobalPerformanceReport, CapabilityStats


class PerformanceTracker:

    def __init__(self):
        self._ready = False
        self._lock  = asyncio.Lock()
        self._conn  = None

    async def initialize(self):
        async with self._lock:
            if self._ready:
                return
            await asyncio.get_event_loop().run_in_executor(None, self._init_db)
            self._ready = True

    def _init_db(self):
        import sqlite3
        settings.ensure_dirs()
        self._conn = sqlite3.connect(str(settings.SQLITE_PATH), check_same_thread=False)
        c = self._conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS agent_performance (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                capability    TEXT NOT NULL,
                domain        TEXT,
                quality_score REAL,
                latency_ms    REAL,
                query_id      TEXT,
                timestamp     TEXT
            );
            CREATE TABLE IF NOT EXISTS feedback_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id    TEXT NOT NULL,
                score           REAL,
                correction_text TEXT,
                correction_type TEXT,
                timestamp       TEXT
            );
            CREATE TABLE IF NOT EXISTS capability_thresholds (
                capability  TEXT PRIMARY KEY,
                threshold   REAL NOT NULL,
                updated_at  TEXT
            );
            CREATE TABLE IF NOT EXISTS execution_summary (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id    TEXT NOT NULL,
                query           TEXT,
                agent_count     INTEGER,
                total_latency   REAL,
                coherence_score REAL,
                timestamp       TEXT
            );
        """)
        self._conn.commit()

    async def record_agent_output(self, capability: str, domain: str,
                                   quality_score: float, latency_ms: float, query_id: str):
        if not self._ready:
            return
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._insert_performance(capability, domain, quality_score, latency_ms, query_id)
        )

    def _insert_performance(self, capability, domain, quality_score, latency_ms, query_id):
        self._conn.execute(
            "INSERT INTO agent_performance (capability, domain, quality_score, latency_ms, query_id, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (capability, domain, quality_score, latency_ms, query_id, datetime.utcnow().isoformat())
        )
        self._conn.commit()

    async def record_feedback(self, execution_id: str, score: float,
                               correction_text: str = None, correction_type: str = "confirm"):
        if not self._ready:
            return
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._insert_feedback(execution_id, score, correction_text, correction_type)
        )

    def _insert_feedback(self, execution_id, score, correction_text, correction_type):
        self._conn.execute(
            "INSERT INTO feedback_log (execution_id, score, correction_text, correction_type, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (execution_id, score, correction_text, correction_type, datetime.utcnow().isoformat())
        )
        self._conn.commit()

    async def record_execution(self, execution_id: str, query: str, agent_count: int,
                                total_latency: float, coherence_score: float):
        if not self._ready:
            return
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._conn.execute(
                "INSERT INTO execution_summary (execution_id, query, agent_count, total_latency, coherence_score, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (execution_id, query, agent_count, total_latency, coherence_score, datetime.utcnow().isoformat())
            ) or self._conn.commit()
        )

    async def get_report(self, global_store=None):
        if not self._ready:
            return GlobalPerformanceReport(
                total_queries=0, avg_quality_score=0.0, capability_stats={},
                cache_hit_rate=0.0, avg_latency_ms=0.0, feedback_count=0,
                intent_library_size=0,
            )
        rows = await asyncio.get_event_loop().run_in_executor(None, self._fetch_report_data)
        total_q, avg_q, cap_stats, avg_lat, fb_count = rows

        intent_size = 0
        if global_store:
            intent_size = await global_store.count_intents()

        return GlobalPerformanceReport(
            total_queries=total_q,
            avg_quality_score=avg_q,
            capability_stats=cap_stats,
            cache_hit_rate=0.0,
            avg_latency_ms=avg_lat,
            feedback_count=fb_count,
            intent_library_size=intent_size,
        )

    def _fetch_report_data(self):
        c = self._conn.cursor()

        c.execute("SELECT COUNT(*) FROM execution_summary")
        total_q = c.fetchone()[0]

        c.execute("SELECT AVG(score) FROM feedback_log")
        row = c.fetchone()
        avg_q = row[0] or 0.0

        c.execute("""
            SELECT capability, AVG(quality_score), AVG(latency_ms), COUNT(*)
            FROM agent_performance GROUP BY capability
        """)
        cap_rows = c.fetchall()
        cap_stats = {}
        for cap, aq, al, cnt in cap_rows:
            cap_stats[cap] = CapabilityStats(
                capability=cap,
                avg_quality=aq or 0.0,
                avg_latency_ms=al or 0.0,
                invocation_count=cnt,
            )

        c.execute("SELECT AVG(total_latency) FROM execution_summary")
        row = c.fetchone()
        avg_lat = row[0] or 0.0

        c.execute("SELECT COUNT(*) FROM feedback_log")
        fb_count = c.fetchone()[0]

        return total_q, avg_q, cap_stats, avg_lat, fb_count

    async def get_threshold(self, capability: str):
        if not self._ready:
            return settings.ACTIVATION_THRESHOLD
        row = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._conn.execute(
                "SELECT threshold FROM capability_thresholds WHERE capability = ?", (capability,)
            ).fetchone()
        )
        return row[0] if row else settings.ACTIVATION_THRESHOLD


# ── Singleton ─────────────────────────────────────────────────────────────
_tracker = None

def get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = PerformanceTracker()
    return _tracker
