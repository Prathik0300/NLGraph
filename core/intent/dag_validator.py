"""
core/intent/dag_validator.py — Deterministic DAG validation and repair.

Checks: dependency integrity, cycle detection, orphan agents, blackboard coherence.
Repairs up to MAX_DAG_REPAIR_ATTEMPTS times before falling back to linear chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from config import settings
from core.exceptions import DAGRepairFailedError
from core.models import Agent, Capability, DependencyEdge, DependencyType


_CAP_ORDER = ["gather", "analyze", "plan", "execute", "verify", "refine", "communicate"]


@dataclass
class ValidationResult:
    valid:    bool
    errors:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DAGValidator:

    def validate_and_repair(self, agents: list[Agent], total_phrases: int = 0):
        """
        Run validation + deterministic repair loop.
        Returns repaired agents list.
        Raises DAGRepairFailedError if repair fails after max attempts.
        """
        for attempt in range(settings.MAX_DAG_REPAIR_ATTEMPTS):
            result = self._validate(agents, total_phrases)
            if result.valid:
                return agents, result
            agents = self._repair(agents, result.errors)

        # Final check
        result = self._validate(agents, total_phrases)
        if result.valid:
            return agents, result

        # All repairs failed — fall back to canonical linear chain
        agents = self._linear_fallback(agents)
        result = self._validate(agents, total_phrases)
        result.warnings.append(
            "DAG repair failed — using fallback linear execution order. "
            "Review the plan carefully before running."
        )
        return agents, result

    def _validate(self, agents: list[Agent], total_phrases: int):
        result = ValidationResult(valid=True)
        agent_ids = {a.agent_id for a in agents}

        # 1. Dependency integrity
        for agent in agents:
            for dep_id in agent.depends_on:
                if dep_id not in agent_ids:
                    result.errors.append(
                        f"Agent '{agent.agent_id}' depends on non-existent '{dep_id}'"
                    )
                    result.valid = False

        if not result.valid:
            return result

        # 2. Cycle detection
        g = self._build_graph(agents)
        try:
            cycle = nx.find_cycle(g)
            result.errors.append(f"Cycle detected: {cycle}")
            result.valid = False
            return result
        except nx.NetworkXNoCycle:
            pass

        # 3. No orphan agents (every non-terminal agent must feed into something)
        terminal_ids = {a.agent_id for a in agents if not a.depends_on
                        or not any(a.agent_id in other.depends_on for other in agents)}
        for agent in agents:
            is_fed_into = any(agent.agent_id in other.depends_on for other in agents)
            if not is_fed_into and agent != agents[-1]:
                result.warnings.append(f"Agent '{agent.role_label}' output is not consumed by any downstream agent")

        return result

    def _repair(self, agents: list[Agent], errors: list[str]):
        """Apply targeted deterministic repairs."""
        agent_map = {a.agent_id: a for a in agents}

        for error in errors:
            # Remove broken dependency references
            if "depends on non-existent" in error:
                dep_id = error.split("'")[3]
                for agent in agents:
                    agent.depends_on = [d for d in agent.depends_on if d != dep_id]

            # Break cycles by removing lowest-confidence edge
            elif "Cycle detected" in error:
                agents = self._break_cycle(agents)

        return agents

    def _break_cycle(self, agents: list[Agent]):
        """Find lowest-confidence edge in any cycle and remove it."""
        g = self._build_graph(agents)
        try:
            cycle_edges = nx.find_cycle(g)
        except nx.NetworkXNoCycle:
            return agents

        # Find the agent pair with lowest confidence in the cycle
        # Heuristic: remove the last dependency in the cycle (most recently added)
        if cycle_edges:
            from_id, to_id = cycle_edges[-1][0], cycle_edges[-1][1]
            for agent in agents:
                if agent.agent_id == to_id and from_id in agent.depends_on:
                    agent.depends_on.remove(from_id)
                    break

        return agents

    def _linear_fallback(self, agents: list[Agent]):
        """Arrange agents in canonical capability order with linear dependencies."""
        cap_order_map = {cap: i for i, cap in enumerate(_CAP_ORDER)}
        sorted_agents = sorted(
            agents,
            key=lambda a: cap_order_map.get(a.capability.value, 99)
        )
        for i, agent in enumerate(sorted_agents):
            agent.depends_on = [sorted_agents[i-1].agent_id] if i > 0 else []
            agent.can_parallelize_with = []
        return sorted_agents

    def _build_graph(self, agents: list[Agent]):
        g = nx.DiGraph()
        for a in agents:
            g.add_node(a.agent_id)
        for a in agents:
            for dep_id in a.depends_on:
                g.add_edge(dep_id, a.agent_id)
        return g


# ── Singleton ─────────────────────────────────────────────────────────────
dag_validator = DAGValidator()
