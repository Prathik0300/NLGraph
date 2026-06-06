"""
core/execution/orchestrator.py — Main query orchestration controller.

The full pipeline per query:
  1. Embed query (dense + sparse via bge-m3)
  2. Check global store for cache hit
  3. Split into phrases (spaCy + benepar + PDTB)
  4. Project onto capability subspaces (multi-prototype + SetFit + cross-encoder)
  5. Detect dependencies (3-layer cascade)
  6. Inject implicit capabilities
  7. Construct agents
  8. Validate + repair DAG
  9. Show user preview (via callbacks)
 10. Execute DAG generations (asyncio.gather per parallel group)
 11. Aggregate results
 12. Store in global memory
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timedelta

from config import settings
from core.exceptions import (
    NLGraphError, PreviewExpiredError, PreviewNotFoundError,
    OllamaUnavailableError,
)
from core.models import (
    Agent, AgentEdit, DecompositionMethod, ExecutionPreview,
    ExecutionResult, FeedbackSubmission, GlobalPerformanceReport,
    IntentMap,
)
from core.intent.embedding_engine import embed_engine
from core.intent.phrase_splitter import phrase_splitter
from core.intent.capability_projector import get_projector
from core.intent.dependency_detector import dependency_detector
from core.intent.implicit_detector import implicit_detector
from core.intent.agent_constructor import agent_constructor, detect_domain
from core.intent.dag_validator import dag_validator
from core.blackboard.blackboard import Blackboard
from core.blackboard.context_builder import context_builder
from core.execution.agent_executor import AgentExecutor
from core.execution.result_aggregator import result_aggregator
from core.memory.global_store import get_global_store
from core.memory.performance_tracker import get_tracker
from core.memory.feedback_collector import feedback_collector


# Preview TTL: 10 minutes
_PREVIEW_TTL = timedelta(minutes=10)


class Orchestrator:

    def __init__(self):
        self._previews: dict[str, tuple[ExecutionPreview, IntentMap]] = {}
        self._executor  = AgentExecutor()
        self._ready     = False

    async def initialize(self):
        # Initialize all singletons
        await embed_engine.initialize()

        global_store = get_global_store()
        await global_store.initialize()

        tracker = get_tracker()
        await tracker.initialize()

        # Load per-capability thresholds from SQLite
        projector = get_projector()
        for cap_val in ["gather", "analyze", "plan", "execute", "verify", "refine", "communicate"]:
            threshold = await tracker.get_threshold(cap_val)
            projector.set_threshold(cap_val, threshold)

        self._ready = True

    # ── submit ────────────────────────────────────────────────────────────

    async def submit(self, query: str, show_internals_callback=None):
        """
        Run the full decomposition pipeline and return an ExecutionPreview.
        Does NOT execute — waits for user confirmation.
        """
        if not self._ready:
            await self.initialize()

        query_id   = str(uuid.uuid4())
        global_store = get_global_store()
        projector    = get_projector()

        # ── Step 1: Embed the query ───────────────────────────────────────
        query_vector = await embed_engine.embed(query)
        hybrid       = await embed_engine.embed_hybrid(query)

        # ── Step 2: Cache hit check ───────────────────────────────────────
        cached = await global_store.lookup_intent(hybrid)
        if cached:
            similarity, plan_json, _ = cached
            import json
            try:
                plan = IntentMap.model_validate_json(plan_json)
                plan.decomposition_method = DecompositionMethod.CACHE_HIT
                plan.cache_similarity     = similarity
                plan.query_id             = query_id
                return self._make_preview(query_id, query, plan)
            except Exception:
                pass  # corrupted cache entry — proceed with full pipeline

        # ── Step 3: Phrase splitting ──────────────────────────────────────
        try:
            phrases = phrase_splitter.split(query)
        except Exception:
            phrases = []

        if show_internals_callback and phrases:
            show_internals_callback("phrases", {"phrases": phrases})

        # ── Step 4: Embed phrases ─────────────────────────────────────────
        import numpy as np
        if phrases:
            phrase_texts  = [p.text for p in phrases]
            phrase_matrix = await embed_engine.embed_batch(phrase_texts)
        else:
            phrase_matrix = query_vector.reshape(1, -1)

        # ── Step 5: Capability projection ─────────────────────────────────
        projection = projector.project(query_vector, phrase_matrix, query)

        if show_internals_callback:
            show_internals_callback("projection", {
                "scores":    projection.scores,
                "threshold": settings.ACTIVATION_THRESHOLD,
            })

        # ── Step 6: Implicit detection ────────────────────────────────────
        implicit = implicit_detector.detect(
            set(projection.active_capabilities), query
        )

        # ── Step 7: Dependency detection ──────────────────────────────────
        # Get the spaCy doc (already built by phrase_splitter if loaded)
        spacy_doc = None
        try:
            if phrase_splitter._nlp is not None:
                spacy_doc = phrase_splitter._nlp(query)
        except Exception:
            pass

        all_caps = list(projection.active_capabilities) + [i.capability.value for i in implicit]
        edges    = dependency_detector.detect(phrases, all_caps, spacy_doc)

        if show_internals_callback:
            show_internals_callback("dependencies", {"edges": edges})

        # ── Step 8: Domain detection ──────────────────────────────────────
        domain = detect_domain(query)

        # ── Step 9: Build agents ──────────────────────────────────────────
        agents = agent_constructor.construct(
            projection=projection,
            implicit=implicit,
            edges=edges,
            phrases=phrases,
            query_id=query_id,
            domain=domain,
            original_query=query,
        )

        if not agents:
            # Vague query — create a single general agent
            from core.models import Capability
            agents = [Agent(
                agent_id=f"general_{query_id[:8]}_00",
                capability=Capability.COMMUNICATE,
                role_label="General Assistant",
                objective=query,
                confidence=0.50,
                llm_model=settings.OLLAMA_LLM_MODEL,
            )]

        # ── Step 10: Validate + repair DAG ───────────────────────────────
        agents, val_result = dag_validator.validate_and_repair(agents, len(phrases))

        # ── Step 11: Build IntentMap ──────────────────────────────────────
        intent_map = IntentMap(
            query_id=query_id,
            original_query=query,
            agents=agents,
            edges=edges,
            activation_scores=projection.scores,
            implicit_additions=implicit,
            total_phrases=len(phrases),
            decomposition_method=DecompositionMethod.VECTOR,
            detected_domain=domain,
        )

        preview = self._make_preview(query_id, query, intent_map, val_result.warnings)

        # Store preview for confirm()
        self._previews[preview.preview_id] = (preview, intent_map, hybrid)
        self._cleanup_expired_previews()

        return preview

    # ── confirm ───────────────────────────────────────────────────────────

    async def confirm(
        self, preview_id: str, edits=None,
        agent_start_callback=None, agent_done_callback=None,
        agent_fail_callback=None,  agent_skip_callback=None,
        generation_start_callback=None, question_callback=None,
    ):
        """
        Execute the plan (with optional user edits applied).
        Returns ExecutionResult.
        """
        if preview_id not in self._previews:
            raise PreviewNotFoundError(f"Preview '{preview_id}' not found or expired.")

        preview, intent_map, hybrid = self._previews.pop(preview_id)

        # Check TTL
        if datetime.utcnow() > preview.expires_at:
            raise PreviewExpiredError(f"Preview '{preview_id}' has expired. Re-submit the query.")

        # Apply edits — write the result back into intent_map so that
        # parallel_groups() (which reads intent_map.agents) sees the full
        # edited agent list, including any user-added agents.
        agents = list(intent_map.agents)
        if edits:
            agents = self._apply_edits(agents, edits)
        intent_map.agents = agents          # keep intent_map in sync

        execution_id = str(uuid.uuid4())
        t_start      = time.perf_counter()
        global_store = get_global_store()
        tracker      = get_tracker()

        # Create blackboard
        blackboard = Blackboard(
            query_id=intent_map.query_id,
            original_query=intent_map.original_query,
            intent_map=intent_map,
        )

        # Execute topological generations (now reflects any added/removed agents)
        all_outputs = []
        parallel_groups = intent_map.parallel_groups()

        for gen_num, generation in enumerate(parallel_groups, start=1):
            gen_agents = [a for a in generation if a.agent_id in {ag.agent_id for ag in agents}]
            if not gen_agents:
                continue

            if generation_start_callback:
                generation_start_callback(gen_num, len(gen_agents))

            tasks = []
            for agent in gen_agents:
                if agent_start_callback:
                    agent_start_callback(agent, gen_num)
                tasks.append(self._run_agent_safe(
                    agent, blackboard, global_store,
                    agent_done_callback, agent_fail_callback, agent_skip_callback,
                ))

            outputs = await asyncio.gather(*tasks)
            all_outputs.extend([o for o in outputs if o is not None])

            # Check for blocking open questions
            if blackboard.open_questions and question_callback:
                answer = await question_callback(blackboard.open_questions[-1])
                if answer:
                    await blackboard.add_fact(
                        __import__('core.models', fromlist=['Fact']).Fact(
                            content=f"Clarification: {answer}",
                            source_agent="user",
                            confidence=1.0,
                        )
                    )

        # Aggregate result
        result = result_aggregator.aggregate(
            outputs=all_outputs,
            preview_id=preview_id,
            query=intent_map.original_query,
            execution_id=execution_id,
            start_time=t_start,
        )
        result.open_questions = list(blackboard.open_questions)

        # Store in global memory (non-blocking)
        asyncio.create_task(self._store_execution(
            result, intent_map, hybrid, global_store, tracker
        ))

        return result

    async def _run_agent_safe(
        self, agent, blackboard, global_store,
        done_cb, fail_cb, skip_cb,
    ):
        try:
            output = await self._executor.execute(
                agent=agent,
                blackboard=blackboard,
                global_store=global_store,
                embed_engine=embed_engine,
            )
            if done_cb:
                done_cb(agent, output)
            return output
        except NLGraphError as e:
            if fail_cb:
                fail_cb(agent, str(e))
            return None
        except Exception as e:
            if fail_cb:
                fail_cb(agent, str(e))
            return None

    # ── feedback ──────────────────────────────────────────────────────────

    async def feedback(self, execution_id: str, score: float, correction: str = None):
        submission = FeedbackSubmission(
            execution_id=execution_id,
            score=score,
            correction=correction,
            correction_type="confirm" if score >= 3.0 else "reject",
        )
        await feedback_collector.record(submission)
        tracker = get_tracker()
        await tracker.record_feedback(
            execution_id=execution_id,
            score=score,
            correction_text=correction,
            correction_type=submission.correction_type,
        )

    # ── stats ─────────────────────────────────────────────────────────────

    async def get_stats(self):
        tracker      = get_tracker()
        global_store = get_global_store()
        return await tracker.get_report(global_store)

    # ── helpers ───────────────────────────────────────────────────────────

    def _make_preview(self, query_id: str, query: str, intent_map: IntentMap, warnings=None):
        parallel_count = sum(
            1 for g in intent_map.parallel_groups() if len(g) > 1
        )
        return ExecutionPreview(
            query=query,
            intent_map=intent_map,
            estimated_agent_count=len(intent_map.agents),
            estimated_parallel_groups=parallel_count,
            confidence_summary=intent_map.activation_scores,
            warnings=warnings or [],
            expires_at=datetime.utcnow() + _PREVIEW_TTL,
        )

    def _apply_edits(self, agents: list[Agent], edits: list[AgentEdit]):
        from core.models import Capability
        from core.intent.agent_constructor import get_role_label, get_llm_model, detect_domain

        for edit in edits:
            if edit.action == "remove" and edit.agent_id:
                agents = [a for a in agents if a.agent_id != edit.agent_id]

            elif edit.action == "add":
                cap_val  = edit.payload.get("capability", "").lower()
                try:
                    cap = Capability(cap_val)
                except ValueError:
                    continue  # unknown capability — skip silently

                # Derive domain + role from existing agents if available
                query    = edit.payload.get("query", "")
                domain   = detect_domain(query) if query else "general"
                role     = get_role_label(cap_val, domain)
                model    = get_llm_model(cap_val)
                obj      = edit.payload.get("objective", query or cap_val)

                # Depend on the last agent in the current list by default
                depends  = [agents[-1].agent_id] if agents else []

                new_agent = Agent(
                    capability   = cap,
                    role_label   = role,
                    objective    = obj,
                    depends_on   = depends,
                    confidence   = 0.85,
                    llm_model    = model,
                )
                agents = agents + [new_agent]

        return agents

    def _cleanup_expired_previews(self):
        now = datetime.utcnow()
        expired = [pid for pid, (preview, _, _) in self._previews.items()
                   if now > preview.expires_at]
        for pid in expired:
            del self._previews[pid]

    async def _store_execution(self, result, intent_map, hybrid, global_store, tracker):
        try:
            await tracker.record_execution(
                execution_id=result.execution_id,
                query=result.query,
                agent_count=len(result.outputs),
                total_latency=result.total_latency_ms,
                coherence_score=result.coherence_score,
            )
            # Store intent vector (score unknown until feedback — store without score)
            await global_store.store_intent(
                query=intent_map.original_query,
                hybrid=hybrid,
                plan=intent_map,
            )
        except Exception:
            pass
