"""
setup/dependency_seeds.py — Seed phrases for the 4 dependency signal types.

These are embedded and averaged to build dependency centroid vectors.
Used in Layer 3 of dependency detection (cross-encoder fallback).

DEPENDENCY_CONTRASTIVE_SEEDS at the bottom provides hard negatives for
validation — phrases that look like one dependency type but belong to another.
Most confusable pairs:
  SEQUENTIAL   ↔ PREREQUISITE  (both imply ordering)
  PREREQUISITE ↔ FEEDS_INTO    (both imply data/knowledge dependency)
  SEQUENTIAL   ↔ FEEDS_INTO    (time-order vs data-flow are easy to conflate)
  PARALLEL     ↔ any           (independent tasks vs dependent ones)
"""

from core.models import DependencyType

DEPENDENCY_SEEDS: dict[str, list[str]] = {

    DependencyType.SEQUENTIAL.value: [
        "then",
        "after that",
        "next",
        "following this",
        "once complete",
        "subsequently",
        "afterward",
        "then do",
        "after completing",
        "once that is done",
        "following that",
        "after finishing",
        "then proceed to",
        "next step is",
        "once this is ready",
        "when done",
        "after that step",
        "in the next step",
        "then move on to",
        "step by step",
        "one after another",
        "in sequence",
        "first then second",
        "do A then B",
        "complete A before B",
        "A then B",
        "before moving to",
        "sequentially",
        "in order",
        "one at a time",
    ],

    DependencyType.PREREQUISITE.value: [
        "based on that",
        "using this knowledge",
        "to inform",
        "which will guide",
        "informed by",
        "grounded in",
        "requires",
        "building on",
        "depends on",
        "given what we know",
        "with this foundation",
        "leveraging the findings",
        "using the insights from",
        "based on the analysis",
        "only after understanding",
        "only possible after",
        "requires first completing",
        "needs the output of",
        "contingent on",
        "predicated on",
        "which requires",
        "which depends on",
        "in order to do B we must first do A",
        "can only start after",
        "must complete first",
        "before this can happen",
    ],

    DependencyType.PARALLEL.value: [
        "at the same time",
        "simultaneously",
        "also",
        "in parallel",
        "alongside",
        "concurrently",
        "while also",
        "at the same time as",
        "both happening together",
        "run at the same time",
        "do both simultaneously",
        "in parallel with",
        "can happen simultaneously",
        "together with",
        "at the same time we should",
        "also doing",
        "meanwhile",
        "while this is happening",
        "independently of",
        "can run concurrently",
        "no dependency between",
        "can be done at same time",
        "simultaneously with",
        "both at once",
    ],

    DependencyType.FEEDS_INTO.value: [
        "to use in",
        "for building",
        "which feeds into",
        "as input to",
        "to apply in",
        "the output of which will be used",
        "using the results of",
        "whose output goes to",
        "providing input for",
        "which becomes the input of",
        "so that it can be used",
        "to feed into",
        "as the basis for",
        "which will inform",
        "the output of A is the input of B",
        "A produces what B needs",
        "directly feeding",
        "used as the foundation for",
        "providing the data for",
        "whose results drive",
        "which directly informs",
        "the product of which is consumed by",
    ],
}


# ── Contrastive seeds (NOT included in centroid mean — used for validation only) ─
#
# Hard negatives per dependency type: phrases superficially similar to that
# type but actually signalling a different dependency relationship.
#
DEPENDENCY_CONTRASTIVE_SEEDS: dict[str, dict[str, list[str]]] = {

    # ── SEQUENTIAL ───────────────────────────────────────────────────────
    # Core signal: pure time-ordering ("do A, then do B")
    # Hard negatives: things that also imply ordering but via data/knowledge
    DependencyType.SEQUENTIAL.value: {
        "not_sequential": [
            # PREREQUISITE (hardest confusable — both imply A before B)
            "using the insights from the previous step",
            "based on what we learned from the analysis",
            "informed by the gathered data",
            "building on the findings",
            "requires that analysis to be complete first",
            "grounded in the earlier research",
            "leveraging the knowledge we just built",
            "contingent on the prior output",
            "predicated on understanding the results first",
            "can only proceed once we understand the data",
            # FEEDS_INTO (hard confusable — both describe A→B flow)
            "whose output becomes the input for the next task",
            "producing what the following step needs",
            "providing the data that drives the next stage",
            "A produces what B consumes",
            "the results of this directly power the next task",
            "feeding its output into the downstream step",
            # PARALLEL (clearly different — no ordering at all)
            "at the same time",
            "simultaneously with",
            "in parallel",
            "concurrently",
            "alongside each other",
            "both at once with no ordering",
        ]
    },

    # ── PREREQUISITE ─────────────────────────────────────────────────────
    # Core signal: knowledge/context dependency ("need to understand X before Y")
    # Hard negatives: pure time-ordering or data-piping, which also imply "A first"
    DependencyType.PREREQUISITE.value: {
        "not_prerequisite": [
            # SEQUENTIAL (hardest confusable — both mean "A before B")
            "then do this",
            "after completing that step",
            "next in the sequence",
            "subsequently",
            "in the next step",
            "once that is done, proceed",
            "do A then B",
            "one after another",
            "step by step",
            "move on to this afterward",
            # FEEDS_INTO (hard confusable — both involve data dependency)
            "whose output is used directly as input",
            "providing the raw data for the next task",
            "A produces what B needs as input",
            "the results of A are consumed by B",
            "feeding the processed data into the model",
            "whose output goes directly to the next stage",
            # PARALLEL
            "simultaneously",
            "at the same time",
            "can be done in parallel",
            "no dependency between them",
            "independently of each other",
        ]
    },

    # ── PARALLEL ─────────────────────────────────────────────────────────
    # Core signal: independence — tasks can run at the same time
    # Hard negatives: any dependency type implies ordering/coupling
    DependencyType.PARALLEL.value: {
        "not_parallel": [
            # SEQUENTIAL (clearly different)
            "then proceed to",
            "after that step is done",
            "one after another",
            "in sequence",
            "complete this before starting that",
            "first A, then B",
            "step by step",
            "do not start B until A is finished",
            # PREREQUISITE
            "depends on",
            "requires first completing",
            "only after understanding",
            "must be done before this can start",
            "contingent on",
            "building on the prior output",
            "informed by the earlier work",
            # FEEDS_INTO
            "whose output goes to",
            "providing input for",
            "which feeds into",
            "A produces what B needs",
            "the results of this drive the next step",
            "feeding data into the downstream task",
        ]
    },

    # ── FEEDS_INTO ───────────────────────────────────────────────────────
    # Core signal: explicit data/artefact flow from A's output to B's input
    # Hard negatives: temporal ordering and knowledge dependency which also
    # describe A→B but without the concrete artefact-passing semantics
    DependencyType.FEEDS_INTO.value: {
        "not_feeds_into": [
            # SEQUENTIAL (hardest confusable — both describe A before B)
            "then do this",
            "after completing that",
            "next in the sequence",
            "step by step",
            "subsequently",
            "do A then do B",
            "one after another in order",
            "complete A before moving to B",
            # PREREQUISITE (hard confusable — both describe data/knowledge flow)
            "informed by",
            "building on those findings",
            "grounded in the prior analysis",
            "using the knowledge gained earlier",
            "leveraging the insights from",
            "based on what was learned",
            "requires understanding the results first",
            "contingent on having analysed the data",
            # PARALLEL
            "simultaneously",
            "at the same time",
            "in parallel",
            "concurrently",
            "no dependency between these tasks",
            "can be done independently",
        ]
    },
}
