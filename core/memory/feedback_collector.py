"""
core/memory/feedback_collector.py — Stores user feedback and triggers learning.
"""

from __future__ import annotations

from core.models import FeedbackSubmission, IntentMap
from core.memory.performance_tracker import get_tracker
from core.memory.global_store import get_global_store


class FeedbackCollector:

    async def record(self, submission: FeedbackSubmission, intent_map: IntentMap = None,
                      hybrid_vector=None):
        tracker      = get_tracker()
        global_store = get_global_store()

        # Always write to feedback log
        await tracker.record_feedback(
            execution_id=submission.execution_id,
            score=submission.score,
            correction_text=submission.correction,
            correction_type=submission.correction_type,
        )

        # Reinforce good plans in global memory
        if submission.score >= 4.0 and intent_map and hybrid_vector:
            await global_store.store_intent(
                query=intent_map.original_query,
                hybrid=hybrid_vector,
                plan=intent_map,
                feedback_score=submission.score,
            )


# ── Singleton ─────────────────────────────────────────────────────────────
feedback_collector = FeedbackCollector()
