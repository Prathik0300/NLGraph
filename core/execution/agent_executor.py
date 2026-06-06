"""
core/execution/agent_executor.py — Runs a single agent via the LLM provider.
"""

from __future__ import annotations

import asyncio
import time
import uuid

from config import settings
from core.exceptions import AgentTimeoutError, AgentExecutionError
from core.models import (
    Agent, AgentContext, AgentOutput, AgentState, AgentStatus,
    Artifact, Fact,
)
from core.blackboard.blackboard import Blackboard
from core.blackboard.context_builder import context_builder
from core.execution.llm_providers.ollama import OllamaProvider


class AgentExecutor:

    def __init__(self, llm_provider=None):
        self._provider       = llm_provider or OllamaProvider()
        self._available_models: list[str] = []   # cached at first use

    async def execute(self, agent: Agent, blackboard: Blackboard,
                       global_store=None, embed_engine=None):
        """
        Execute a single agent:
        1. Build context from blackboard
        2. Call LLM
        3. Write output to blackboard
        Returns AgentOutput.
        """
        t_start = time.perf_counter()

        # Mark running
        await blackboard.update_agent_state(agent.agent_id, AgentState(
            agent_id=agent.agent_id,
            status=AgentStatus.RUNNING,
            started_at=__import__('datetime').datetime.utcnow(),
        ))
        await blackboard.log("agent_start", agent_id=agent.agent_id)

        try:
            # Build context
            context = await context_builder.build(agent, blackboard, global_store, embed_engine)
            prompt  = context_builder.format_prompt(agent, context, blackboard.original_query)

            # Call LLM with timeout.
            # If the capability-specific model isn't pulled, fall back to
            # OLLAMA_FALLBACK_MODEL (mistral:7b by default) automatically.
            # Available model list is fetched once per executor instance.
            llm_model = agent.llm_model or settings.OLLAMA_LLM_MODEL
            if not self._available_models:
                self._available_models = await self._provider.list_models()
            if self._available_models and not any(
                llm_model in m or m in llm_model for m in self._available_models
            ):
                llm_model = settings.OLLAMA_FALLBACK_MODEL

            try:
                content = await asyncio.wait_for(
                    self._provider.generate(
                        prompt=prompt,
                        model=llm_model,
                        system=f"You are a {agent.role_label}. Be precise and focused.",
                    ),
                    timeout=settings.OLLAMA_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                raise AgentTimeoutError(agent.agent_id, settings.OLLAMA_TIMEOUT_SECONDS)

            latency_ms = (time.perf_counter() - t_start) * 1000

            # Write artifact to blackboard
            artifact_id  = str(uuid.uuid4())
            artifact_key = f"{agent.capability.value}_output"
            artifact     = Artifact(
                artifact_id=artifact_id,
                artifact_key=artifact_key,
                content=content,
                produced_by=agent.agent_id,
                format="text",
            )
            await blackboard.add_artifact(artifact)

            # Extract facts from output (simple heuristic: each sentence is a fact)
            if embed_engine is not None:
                await self._extract_facts(content, agent.agent_id, blackboard, embed_engine)

            # Mark complete
            await blackboard.update_agent_state(agent.agent_id, AgentState(
                agent_id=agent.agent_id,
                status=AgentStatus.COMPLETE,
                started_at=__import__('datetime').datetime.utcnow(),
                completed_at=__import__('datetime').datetime.utcnow(),
                output_artifact_id=artifact_id,
                latency_ms=latency_ms,
            ))
            await blackboard.log("agent_complete", agent_id=agent.agent_id,
                                  latency_ms=latency_ms)

            return AgentOutput(
                agent_id=agent.agent_id,
                capability=agent.capability,
                content=content,
                latency_ms=latency_ms,
                artifact_id=artifact_id,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - t_start) * 1000
            error_msg  = str(e)

            await blackboard.update_agent_state(agent.agent_id, AgentState(
                agent_id=agent.agent_id,
                status=AgentStatus.FAILED,
                error=error_msg,
                latency_ms=latency_ms,
            ))
            await blackboard.log("agent_failed", agent_id=agent.agent_id,
                                  level="error", error=error_msg)

            # Write placeholder so downstream agents get a clear signal
            await blackboard.add_artifact(Artifact(
                artifact_key=f"{agent.capability.value}_output",
                content=f"[Agent {agent.role_label} failed: {error_msg[:200]}]",
                produced_by=agent.agent_id,
                format="text",
            ))

            raise AgentExecutionError(f"Agent '{agent.role_label}' failed: {error_msg}") from e

    async def _extract_facts(self, content: str, agent_id: str,
                              blackboard: Blackboard, embed_engine):
        """Extract key sentences as facts and add to blackboard with embeddings."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        # Take first 5 non-trivial sentences as facts
        for sent in sentences[:5]:
            sent = sent.strip()
            if len(sent.split()) < 6:
                continue
            try:
                vec = await embed_engine.embed(sent)
                fact = Fact(
                    content=sent,
                    source_agent=agent_id,
                    confidence=0.75,
                    embedding=vec.tolist(),
                )
                await blackboard.add_fact(fact)
            except Exception:
                pass
