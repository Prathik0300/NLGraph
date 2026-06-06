"""
core/intent/dependency_detector.py — 3-layer dependency detection cascade.

Layer 1: PDTB connective lexicon   (<1ms, explicit connectives, highest precision)
Layer 2: spaCy dependency signals  (structural grammar signals)
Layer 3: Cross-encoder classifier  (ambiguous pairs, fires when Layers 1+2 disagree)

Final score = 0.45*lexical + 0.30*parse + 0.25*cross_encoder
Default fallback = SEQUENTIAL when confidence < 0.55 (over-sequencing is safe)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import settings
from core.models import Capability, DependencyEdge, DependencyType, Phrase


# ── PDTB lexicon (re-use same file as phrase splitter) ────────────────────
_PDTB_PATH = Path(__file__).parent.parent.parent / "setup" / "pdtb_connectives.json"

def _load_pdtb_flat():
    if not _PDTB_PATH.exists():
        return {}
    raw = json.loads(_PDTB_PATH.read_text())
    flat = {}
    for cat, entries in raw.items():
        if cat.startswith("_"):
            continue
        for phrase, meta in entries.items():
            flat[phrase.lower()] = meta
    return flat

_PDTB: dict = _load_pdtb_flat()


# ── Capability-pair Bayesian priors ───────────────────────────────────────
# (from_cap, to_cap) -> (dep_type, prior_weight)
_CAP_PRIORS: dict[tuple[str, str], tuple[str, float]] = {
    ("gather",  "analyze"):     ("feeds_into",   0.80),
    ("analyze", "plan"):        ("feeds_into",   0.75),
    ("analyze", "execute"):     ("feeds_into",   0.70),
    ("plan",    "execute"):     ("prerequisite", 0.85),
    ("execute", "verify"):      ("sequential",   0.90),
    ("verify",  "refine"):      ("feeds_into",   0.80),
    ("refine",  "communicate"): ("feeds_into",   0.70),
    ("gather",  "execute"):     ("feeds_into",   0.65),
    ("plan",    "verify"):      ("parallel",     0.60),
}

# Canonical capability ordering for default linear fallback
_CAP_ORDER = ["gather", "analyze", "plan", "execute", "verify", "refine", "communicate"]


# ══════════════════════════════════════════════════════════════════════════
# DEPENDENCY DETECTOR
# ══════════════════════════════════════════════════════════════════════════

class DependencyDetector:

    def __init__(self):
        self._cross_encoder = None

    def _ensure_cross_encoder(self):
        if self._cross_encoder is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(settings.CROSS_ENC_MODEL)
        except Exception:
            self._cross_encoder = None

    def detect(self, phrases: list[Phrase], active_capabilities: list[str], spacy_doc=None):
        """
        Detect dependency edges between active capabilities.

        Returns list of DependencyEdge sorted by (from_cap_order, to_cap_order).
        """
        if len(active_capabilities) < 2:
            return []

        # Order capabilities by canonical order for consistent pairing
        ordered = [c for c in _CAP_ORDER if c in active_capabilities]
        # Add any not in canonical order at the end
        for c in active_capabilities:
            if c not in ordered:
                ordered.append(c)

        edges = []
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b    = ordered[i], ordered[j]
                # Only adjacent or near-adjacent (within 2 positions)
                if j - i > 2:
                    continue
                edge = self._detect_pair(a, b, phrases, spacy_doc)
                if edge is not None:
                    edges.append(edge)

        return edges

    def _detect_pair(self, cap_a: str, cap_b: str, phrases: list[Phrase], spacy_doc=None):
        """Run all three layers for a single (cap_a → cap_b) pair."""

        # Find phrases assigned to each capability (via phrase connective_signal as proxy)
        text_between = self._extract_text_between(cap_a, cap_b, phrases)

        # Layer 1: PDTB lexicon
        l1_scores, l1_type = self._layer1_lexical(text_between)

        # Layer 2: spaCy parse signals
        l2_scores, l2_type = self._layer2_parse(cap_a, cap_b, spacy_doc)

        # Layer 3: cross-encoder (only when layers disagree or confidence low)
        max_l1 = max(l1_scores.values()) if l1_scores else 0.0
        max_l2 = max(l2_scores.values()) if l2_scores else 0.0
        l3_scores = {}

        if max_l1 < 0.65 and max_l2 < 0.65 and text_between:
            try:
                self._ensure_cross_encoder()
                if self._cross_encoder is not None:
                    l3_scores = self._layer3_crossencoder(text_between)
            except Exception:
                pass

        # Capability-pair prior
        prior_scores = self._apply_prior(cap_a, cap_b)

        # Ensemble
        all_types = {t.value for t in DependencyType if t != DependencyType.NONE}
        final_scores = {}
        for dep_type in all_types:
            s1 = l1_scores.get(dep_type, 0.0)
            s2 = l2_scores.get(dep_type, 0.0)
            s3 = l3_scores.get(dep_type, 0.0)
            sp = prior_scores.get(dep_type, 0.0)

            # Weighted combination (prior has small influence)
            w1, w2, w3, wp = 0.40, 0.30, 0.20, 0.10
            if not l3_scores:
                w1, w2, wp = 0.50, 0.40, 0.10
                w3 = 0.0
            final_scores[dep_type] = w1*s1 + w2*s2 + w3*s3 + wp*sp

        # Pick winner
        best_type  = max(final_scores, key=final_scores.get)
        best_score = final_scores[best_type]

        # Fallback to SEQUENTIAL if confidence too low
        if best_score < settings.DEP_MIN_CONFIDENCE:
            best_type  = DependencyType.SEQUENTIAL.value
            best_score = settings.DEP_MIN_CONFIDENCE

        try:
            dep_enum = DependencyType(best_type)
        except ValueError:
            dep_enum = DependencyType.SEQUENTIAL

        return DependencyEdge(
            from_capability=Capability(cap_a),
            to_capability=Capability(cap_b),
            dep_type=dep_enum,
            confidence=min(best_score, 1.0),
            signal_breakdown={
                "lexical":      max(l1_scores.values()) if l1_scores else 0.0,
                "parse":        max(l2_scores.values()) if l2_scores else 0.0,
                "cross_encoder":max(l3_scores.values()) if l3_scores else 0.0,
                "prior":        max(prior_scores.values()) if prior_scores else 0.0,
            },
        )

    # ── Layer 1: PDTB lexical lookup ─────────────────────────────────────

    def _layer1_lexical(self, text_between: str):
        if not text_between:
            return {}, None
        lower = text_between.lower()
        scores = {t.value: 0.0 for t in DependencyType if t != DependencyType.NONE}
        best_conf = 0.0
        best_type = None
        for phrase, meta in _PDTB.items():
            if phrase in lower:
                dep_type = meta.get("type")
                conf     = meta.get("confidence", 0.7)
                if dep_type in scores and conf > scores[dep_type]:
                    scores[dep_type] = conf
                if conf > best_conf:
                    best_conf = conf
                    best_type = dep_type
        return scores, best_type

    # ── Layer 2: spaCy dependency parse signals ───────────────────────────

    def _layer2_parse(self, cap_a: str, cap_b: str, spacy_doc=None):
        scores = {t.value: 0.0 for t in DependencyType if t != DependencyType.NONE}
        if spacy_doc is None:
            return scores, None

        best_type = None
        best_conf = 0.0

        for token in spacy_doc:
            dep   = token.dep_
            head  = token.head

            if dep == "conj" and head.pos_ in ("VERB", "AUX"):
                scores["parallel"]   = max(scores["parallel"], 0.65)
                scores["sequential"] = max(scores["sequential"], 0.55)

            elif dep == "advcl":
                mark_tokens = [c for c in token.children if c.dep_ == "mark"]
                if mark_tokens:
                    mark_text = mark_tokens[0].text.lower()
                    if mark_text in ("after", "once", "when", "then"):
                        scores["sequential"]   = max(scores["sequential"], 0.80)
                    elif mark_text in ("before", "until", "prior"):
                        scores["prerequisite"] = max(scores["prerequisite"], 0.78)
                    elif mark_text in ("while", "as", "during"):
                        scores["parallel"]     = max(scores["parallel"], 0.75)
                    elif mark_text in ("because", "since", "so"):
                        scores["prerequisite"] = max(scores["prerequisite"], 0.72)
                    elif mark_text in ("to", "in order to", "so as to"):
                        scores["feeds_into"]   = max(scores["feeds_into"], 0.77)
                else:
                    scores["sequential"] = max(scores["sequential"], 0.60)

            elif dep in ("relcl", "acl"):
                scores["feeds_into"] = max(scores["feeds_into"], 0.65)

            elif dep == "prep" and token.text.lower() in ("using", "based", "from"):
                scores["feeds_into"]   = max(scores["feeds_into"], 0.68)
                scores["prerequisite"] = max(scores["prerequisite"], 0.60)

        best_type = max(scores, key=scores.get) if any(scores.values()) else None
        best_conf = scores.get(best_type, 0.0) if best_type else 0.0

        return scores, best_type

    # ── Layer 3: cross-encoder ────────────────────────────────────────────

    def _layer3_crossencoder(self, text_between: str):
        if not self._cross_encoder or not text_between:
            return {}
        hypotheses = {
            "sequential":   f"The phrase '{text_between}' indicates A happens before B in sequence.",
            "prerequisite": f"The phrase '{text_between}' indicates B depends on A being completed first.",
            "parallel":     f"The phrase '{text_between}' indicates A and B happen simultaneously.",
            "feeds_into":   f"The phrase '{text_between}' indicates the output of A is used directly by B.",
        }
        try:
            pairs = [("classify this dependency", h) for h in hypotheses.values()]
            raw   = self._cross_encoder.predict(pairs)
            keys  = list(hypotheses.keys())
            return {keys[i]: float(raw[i]) for i in range(len(keys))}
        except Exception:
            return {}

    # ── Prior ─────────────────────────────────────────────────────────────

    def _apply_prior(self, cap_a: str, cap_b: str):
        scores   = {t.value: 0.0 for t in DependencyType if t != DependencyType.NONE}
        key      = (cap_a, cap_b)
        if key in _CAP_PRIORS:
            dep_type, weight = _CAP_PRIORS[key]
            scores[dep_type] = weight
        return scores

    # ── Extract inter-phrase text ─────────────────────────────────────────

    def _extract_text_between(self, cap_a: str, cap_b: str, phrases: list[Phrase]):
        """Get the raw text of any connective phrases between the two capabilities."""
        if not phrases:
            return ""
        texts = [p.text for p in phrases if p.connective_signal]
        return " ".join(texts) if texts else ""


# ── Singleton ─────────────────────────────────────────────────────────────
dependency_detector = DependencyDetector()
