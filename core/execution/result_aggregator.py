"""
core/execution/result_aggregator.py — Compiles agent outputs into ExecutionResult.
"""

from __future__ import annotations

import time
import numpy as np

from core.models import AgentOutput, ExecutionResult


class ResultAggregator:

    def aggregate(self, outputs: list[AgentOutput], preview_id: str,
                   query: str, execution_id: str, start_time: float):
        total_ms = (time.perf_counter() - start_time) * 1000

        # Build final answer by concatenating outputs in execution order
        sections = []
        for out in outputs:
            sections.append(f"[{out.capability.value.upper()}]\n{out.content}")
        final_answer = "\n\n".join(sections)

        # Coherence score: mean pairwise cosine similarity (placeholder)
        coherence = self._coherence_score(outputs)

        return ExecutionResult(
            execution_id=execution_id,
            preview_id=preview_id,
            query=query,
            outputs=outputs,
            final_answer=final_answer,
            coherence_score=coherence,
            total_latency_ms=total_ms,
        )

    def _coherence_score(self, outputs: list[AgentOutput]):
        """Simple heuristic: 1.0 if all agents completed, scaled down by failures."""
        if not outputs:
            return 0.0
        completed = sum(1 for o in outputs if not o.content.startswith("[Agent"))
        return round(completed / len(outputs), 2)


result_aggregator = ResultAggregator()
