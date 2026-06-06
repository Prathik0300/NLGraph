"""
core/blackboard/context_builder.py — Assembles each agent's context window.

Priority order (highest first):
  1. Upstream artifacts      (always included — agent needs predecessors' output)
  2. Semantically relevant facts from blackboard  (top-K by FAISS similarity)
  3. High-quality historical examples from global store  (top-3)
  4. Open questions          (included if budget allows)

Token budget: CONTEXT_TOKEN_BUDGET (default 3000 estimated tokens).
Token estimate: len(text.split()) * 1.3
"""

from __future__ import annotations

from config import settings
from core.models import Agent, AgentContext, Artifact, Fact
from core.blackboard.blackboard import Blackboard


def _token_estimate(text: str):
    return int(len(text.split()) * 1.3)


class ContextBuilder:

    async def build(self, agent: Agent, blackboard: Blackboard, global_store=None, embed_engine=None):
        """
        Assemble the AgentContext for a given agent.
        """
        budget = settings.CONTEXT_TOKEN_BUDGET
        used   = 0

        # ── 1. Upstream artifacts ────────────────────────────────────────
        upstream = blackboard.get_upstream_artifacts(agent)

        # ── 2. Semantic fact retrieval ────────────────────────────────────
        relevant_facts: list[Fact] = []
        if embed_engine is not None:
            try:
                obj_vec = await embed_engine.embed(agent.objective)
                relevant_facts = await blackboard.search_facts(
                    obj_vec,
                    top_k=settings.CONTEXT_TOP_K_FACTS,
                )
            except Exception:
                relevant_facts = blackboard.facts[-settings.CONTEXT_TOP_K_FACTS:]
        else:
            relevant_facts = blackboard.facts[-settings.CONTEXT_TOP_K_FACTS:]

        # ── 3. Global memory examples ────────────────────────────────────
        global_examples: list[str] = []
        if global_store is not None:
            try:
                if embed_engine is not None:
                    qv = await embed_engine.embed(agent.objective)
                    global_examples = await global_store.get_relevant_outputs(
                        capability=agent.capability.value,
                        query_vector=qv,
                        top_k=3,
                    )
            except Exception:
                pass

        # ── 4. Budget management ─────────────────────────────────────────
        # Priority: upstream > facts > global > questions
        selected_upstream   = []
        selected_facts      = []
        selected_global     = []

        for art in upstream:
            tokens = _token_estimate(art.content)
            if used + tokens <= budget:
                selected_upstream.append(art)
                used += tokens
            else:
                # Truncate to ~200 tokens
                truncated = " ".join(art.content.split()[:200])
                selected_upstream.append(Artifact(
                    artifact_id=art.artifact_id,
                    artifact_key=art.artifact_key,
                    content=truncated + " [truncated]",
                    produced_by=art.produced_by,
                    format=art.format,
                ))
                used += _token_estimate(truncated)

        for fact in relevant_facts:
            tokens = _token_estimate(fact.content)
            if used + tokens <= budget:
                selected_facts.append(fact)
                used += tokens

        for ex in global_examples:
            tokens = _token_estimate(ex)
            if used + tokens <= budget:
                selected_global.append(ex)
                used += tokens

        return AgentContext(
            agent_id=agent.agent_id,
            objective=agent.objective,
            relevant_facts=selected_facts,
            upstream_artifacts=selected_upstream,
            global_examples=selected_global,
            open_questions=list(blackboard.open_questions),
            token_count_estimate=used,
        )

    def format_prompt(self, agent: Agent, context: AgentContext, query: str):
        """Build the final prompt string sent to the LLM."""
        parts = [
            f"You are a {agent.role_label}.",
            f"Overall query: {query}",
            f"Your specific objective: {agent.objective}",
            "",
        ]

        if context.upstream_artifacts:
            parts.append("=== Output from preceding agents ===")
            for art in context.upstream_artifacts:
                parts.append(f"[{art.artifact_key}]\n{art.content}")
            parts.append("")

        if context.relevant_facts:
            parts.append("=== Relevant facts from this session ===")
            for fact in context.relevant_facts:
                conf = f"(confidence {fact.confidence:.2f})"
                parts.append(f"• {fact.content}  {conf}")
            parts.append("")

        if context.global_examples:
            parts.append("=== High-quality examples from past sessions ===")
            for ex in context.global_examples:
                parts.append(ex[:400])
            parts.append("")

        if context.open_questions:
            parts.append("=== Open questions (be aware of these) ===")
            for q in context.open_questions:
                parts.append(f"? {q}")
            parts.append("")

        parts.append("Produce your output now:")
        return "\n".join(parts)


# ── Singleton ─────────────────────────────────────────────────────────────
context_builder = ContextBuilder()
