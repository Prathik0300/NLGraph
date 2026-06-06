"""
core/intent/agent_constructor.py — Builds Agent objects from projection results.

Combines: capability activations + implicit additions + dependency edges
→ produces a list of Agent objects ready for DAG construction.
"""

from __future__ import annotations

import uuid

from core.models import (
    Agent, Capability, DependencyEdge, ImplicitAddition,
)
from core.intent.capability_projector import ProjectionResult
from config import settings


# ── Role label lookup: (capability, domain) → role label ─────────────────
_ROLE_LABELS: dict[tuple[str, str], str] = {
    ("gather",      "software"):  "Researcher",
    ("gather",      "research"):  "Literature Reviewer",
    ("gather",      "business"):  "Market Analyst",
    ("gather",      "data"):      "Data Collector",
    ("gather",      "general"):   "Information Gatherer",

    ("analyze",     "software"):  "Synthesizer",
    ("analyze",     "research"):  "Concept Analyst",
    ("analyze",     "business"):  "Business Analyst",
    ("analyze",     "data"):      "Data Analyst",
    ("analyze",     "general"):   "Analyst",

    ("plan",        "software"):  "Architect",
    ("plan",        "research"):  "Research Planner",
    ("plan",        "business"):  "Strategist",
    ("plan",        "general"):   "Planner",

    ("execute",     "software"):  "Coder",
    ("execute",     "research"):  "Author",
    ("execute",     "business"):  "Producer",
    ("execute",     "data"):      "Pipeline Engineer",
    ("execute",     "general"):   "Builder",

    ("verify",      "software"):  "Tester",
    ("verify",      "research"):  "Fact Checker",
    ("verify",      "business"):  "QA Reviewer",
    ("verify",      "general"):   "Verifier",

    ("refine",      "software"):  "Refactorer",
    ("refine",      "research"):  "Editor",
    ("refine",      "business"):  "Optimizer",
    ("refine",      "general"):   "Refiner",

    ("communicate", "software"):  "Documenter",
    ("communicate", "research"):  "Writer",
    ("communicate", "business"):  "Presenter",
    ("communicate", "general"):   "Communicator",
}

# ── Domain keyword detection ──────────────────────────────────────────────
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "software": ["code", "implement", "function", "class", "api", "debug",
                 "deploy", "backend", "frontend", "database", "script", "bug"],
    "research": ["paper", "literature", "hypothesis", "study", "findings",
                 "cite", "journal", "survey", "academic", "research", "theory"],
    "business": ["revenue", "strategy", "market", "customer", "roi",
                 "stakeholder", "business", "client", "sales", "product"],
    "data":     ["data", "dataset", "csv", "sql", "query", "pipeline",
                 "model", "train", "features", "metrics", "statistics"],
    "creative": ["design", "write", "story", "compose", "illustrate",
                 "draft", "content", "creative", "narrative"],
}


def detect_domain(query: str):
    lower  = query.lower()
    scores = {domain: sum(1 for kw in kws if kw in lower)
              for domain, kws in _DOMAIN_KEYWORDS.items()}
    best   = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def get_role_label(capability: str, domain: str):
    return (
        _ROLE_LABELS.get((capability, domain))
        or _ROLE_LABELS.get((capability, "general"))
        or capability.upper()
    )


# ── LLM model routing ────────────────────────────────────────────────────

def get_llm_model(capability: str):
    return settings.CAPABILITY_MODEL_MAP.get(capability, settings.OLLAMA_LLM_MODEL)


# ── Blackboard key templates ──────────────────────────────────────────────

def _writes_key(capability: str):
    return f"{capability}_output"

def _reads_keys(capability: str, depends_on_caps: list[str]):
    return [f"{cap}_output" for cap in depends_on_caps]


# ══════════════════════════════════════════════════════════════════════════
# AGENT CONSTRUCTOR
# ══════════════════════════════════════════════════════════════════════════

class AgentConstructor:

    def construct(
        self,
        projection:   ProjectionResult,
        implicit:     list[ImplicitAddition],
        edges:        list[DependencyEdge],
        phrases,
        query_id:     str,
        domain:       str = "general",
        original_query: str = "",
    ):
        """
        Build a list of Agent objects from the full projection context.
        """
        # Merge active + implicit capabilities
        all_caps: list[tuple[str, float, bool]] = []

        for cap_val in projection.active_capabilities:
            score = projection.scores.get(cap_val, 0.0)
            all_caps.append((cap_val, score, False))

        for imp in implicit:
            all_caps.append((imp.capability.value, imp.confidence, True))

        # Build dependency lookup: cap_value → list[cap_value that precedes it]
        dep_map: dict[str, list[str]] = {cap: [] for cap, _, _ in all_caps}
        parallel_map: dict[str, list[str]] = {cap: [] for cap, _, _ in all_caps}

        for edge in edges:
            frm = edge.from_capability.value
            to  = edge.to_capability.value
            if to in dep_map:
                if edge.dep_type.value in ("sequential", "prerequisite", "feeds_into"):
                    dep_map[to].append(frm)
                elif edge.dep_type.value == "parallel":
                    parallel_map[frm].append(to)
                    parallel_map[to].append(frm)

        # Assign phrases to capabilities
        cap_phrases: dict[str, list[str]] = {cap: [] for cap, _, _ in all_caps}
        if phrases:
            for phrase_idx, cap_val in projection.phrase_assignments.items():
                if cap_val in cap_phrases and phrase_idx < len(phrases):
                    cap_phrases[cap_val].append(phrases[phrase_idx].text)

        agents = []
        for i, (cap_val, score, is_implicit) in enumerate(all_caps):
            agent_id = f"{cap_val}_{query_id[:8]}_{i:02d}"
            dep_caps = dep_map.get(cap_val, [])

            # Resolve dep_caps to agent_ids
            cap_to_aid = {c: f"{c}_{query_id[:8]}_{j:02d}"
                          for j, (c, _, _) in enumerate(all_caps)}
            depends_on = [cap_to_aid[dc] for dc in dep_caps if dc in cap_to_aid]

            par_caps = parallel_map.get(cap_val, [])
            parallels = [cap_to_aid[pc] for pc in par_caps if pc in cap_to_aid]

            # Build objective from assigned phrases.
            # If the phrase-derived text is too short (a fragment) or missing,
            # fall back to the full original query so the agent has proper context.
            obj_phrases = cap_phrases.get(cap_val, [])
            if obj_phrases:
                objective = " and ".join(obj_phrases)
                # Reject fragments: if result is shorter than 4 words, use full query
                if len(objective.split()) < 4 and original_query:
                    objective = original_query
            elif original_query:
                objective = original_query
            else:
                objective = f"Perform {cap_val} tasks for this query"

            agents.append(Agent(
                agent_id=agent_id,
                capability=Capability(cap_val),
                role_label=get_role_label(cap_val, domain),
                objective=objective,
                depends_on=depends_on,
                can_parallelize_with=parallels,
                reads_from_blackboard=_reads_keys(cap_val, dep_caps),
                writes_to_blackboard=[_writes_key(cap_val)],
                confidence=score,
                is_implicit=is_implicit,
                phrase_indices=[
                    idx for idx, cv in projection.phrase_assignments.items()
                    if cv == cap_val
                ],
                llm_model=get_llm_model(cap_val),
            ))

        return agents


# ── Singleton ─────────────────────────────────────────────────────────────
agent_constructor = AgentConstructor()
