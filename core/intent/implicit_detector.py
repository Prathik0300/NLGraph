"""
core/intent/implicit_detector.py — Implicit capability detection via rule engine.

Rules are data, not logic — defined as ImplicationRule objects so they can be
extended without touching the detection logic.

Each rule has:
  - triggers:           capabilities that must be active
  - missing:            capability that must NOT be active
  - confidence:         how certain the implication is
  - suppression_signals: query words that cancel the rule
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.models import Capability, ImplicitAddition


# ══════════════════════════════════════════════════════════════════════════
# IMPLICATION RULE
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ImplicationRule:
    triggers: list[str]          # capability values that must ALL be active
    missing: str                  # capability value that must NOT be active
    confidence: float
    reason: str
    suppression_signals: list[str] = field(default_factory=list)


# ── Rule definitions ──────────────────────────────────────────────────────
IMPLICATION_RULES: list[ImplicationRule] = [
    ImplicationRule(
        triggers=["execute"],
        missing="plan",
        confidence=0.85,
        reason="Execute without Plan implies planning was intended",
        suppression_signals=[
            "existing", "already written", "already built", "this code",
            "current implementation", "the script", "the file",
        ],
    ),
    ImplicationRule(
        triggers=["verify"],
        missing="execute",
        confidence=0.90,
        reason="Verify implies something to build and verify",
        suppression_signals=[
            "existing", "already built", "current output", "this result",
            "review this", "check if", "look at the",
        ],
    ),
    ImplicationRule(
        triggers=["gather", "execute"],
        missing="analyze",
        confidence=0.78,
        reason="Gather + Execute without Analyze implies an analysis step",
        suppression_signals=["quickly", "fast", "simple", "just"],
    ),
    ImplicationRule(
        triggers=["refine"],
        missing="execute",
        confidence=0.72,
        reason="Refine implies something was previously built",
        suppression_signals=["existing", "draft", "current version"],
    ),
    ImplicationRule(
        triggers=["communicate"],
        missing="gather",
        confidence=0.65,
        reason="Writing/presenting often needs research first",
        suppression_signals=[
            "summarize", "existing findings", "the provided", "from the above",
        ],
    ),
]


# ══════════════════════════════════════════════════════════════════════════
# IMPLICIT DETECTOR
# ══════════════════════════════════════════════════════════════════════════

class ImplicitDetector:

    def detect(self, active: set[str], query_text: str = ""):
        """
        Check all implication rules against the current active set.
        Returns a list of ImplicitAddition objects for capabilities to inject.

        Rules fire in order — later rules see capabilities added by earlier rules.
        """
        active  = set(active)   # mutable working copy
        results = []
        lower   = query_text.lower()

        for rule in IMPLICATION_RULES:
            # All trigger capabilities must be active
            if not all(t in active for t in rule.triggers):
                continue
            # Missing capability must NOT already be active
            if rule.missing in active:
                continue
            # Check suppression signals
            suppressed = any(sig in lower for sig in rule.suppression_signals)
            if suppressed:
                continue

            # Rule fires
            addition = ImplicitAddition(
                capability=Capability(rule.missing),
                reason=rule.reason,
                confidence=rule.confidence,
                triggered_by=[Capability(t) for t in rule.triggers],
            )
            results.append(addition)
            active.add(rule.missing)   # let subsequent rules see this addition

        return results


# ── Singleton ─────────────────────────────────────────────────────────────
implicit_detector = ImplicitDetector()
