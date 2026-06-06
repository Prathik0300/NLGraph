"""
core/intent/capability_projector.py — Multi-prototype capability projection.

For each of the 7 capabilities, K=3 prototype centroids are stored in a .npz file.
A query activates a capability if its embedding is close enough to ANY prototype.

3-tier ensemble:
  Tier 1: Multi-prototype cosine projection  (always runs, no training needed)
  Tier 2: SetFit classifier                  (runs when checkpoint exists)
  Tier 3: Cross-encoder re-ranking           (runs only on ambiguous cases)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import settings
from core.exceptions import CentroidNotFoundError, StaleCentroidError, ProjectionError
from core.models import Capability


# ══════════════════════════════════════════════════════════════════════════
# CENTROID STORE
# ══════════════════════════════════════════════════════════════════════════

class CentroidStore:
    """Loads and serves the centroid .npz built by setup/centroid_builder.py."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._data = None

    def load(self):
        if not self._path.exists():
            raise CentroidNotFoundError(
                f"Centroid file not found: {self._path}\n"
                f"Run: python -m setup.centroid_builder"
            )
        self._data = np.load(str(self._path), allow_pickle=False)
        if "_model_name" in self._data:
            stored = bytes(self._data["_model_name"]).decode("utf-8")
            if stored != settings.EMBED_MODEL:
                raise StaleCentroidError(
                    f"Centroids built with '{stored}' but current model is "
                    f"'{settings.EMBED_MODEL}'. Rebuild: python -m setup.centroid_builder"
                )

    def is_loaded(self):
        return self._data is not None

    def get_prototypes(self, capability: str):
        key = f"cap_{capability}_prototypes"
        if self._data is None or key not in self._data:
            raise CentroidNotFoundError(f"No prototypes for: {capability}")
        return self._data[key]

    def all_prototypes(self):
        if self._data is None:
            return {}
        return {
            cap.value: self._data[f"cap_{cap.value}_prototypes"]
            for cap in Capability
            if f"cap_{cap.value}_prototypes" in self._data
        }


# ══════════════════════════════════════════════════════════════════════════
# PROJECTION RESULT
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ProjectionResult:
    active_capabilities: list[str]
    scores: dict[str, float]
    phrase_assignments: dict[int, str]
    method_used: str = "multi_prototype"
    ambiguous_resolved: bool = False


# ══════════════════════════════════════════════════════════════════════════
# CAPABILITY PROJECTOR
# ══════════════════════════════════════════════════════════════════════════

class CapabilityProjector:

    def __init__(self, centroid_store: CentroidStore):
        self._store                              = centroid_store
        self._setfit_model                       = None
        self._cross_encoder                      = None
        self._threshold                          = settings.ACTIVATION_THRESHOLD
        self._per_cap_thresholds: dict[str, float] = {}

    def set_threshold(self, capability: str, value: float):
        self._per_cap_thresholds[capability] = value

    def _get_threshold(self, capability: str):
        return self._per_cap_thresholds.get(capability, self._threshold)

    def load_setfit_if_available(self):
        model_path = settings.SETFIT_MODEL_DIR / "latest"
        if not model_path.exists():
            return
        try:
            from setfit import SetFitModel
            self._setfit_model = SetFitModel.from_pretrained(str(model_path))
        except Exception:
            self._setfit_model = None

    def _ensure_cross_encoder(self):
        if self._cross_encoder is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(settings.CROSS_ENC_MODEL)
        except Exception:
            self._cross_encoder = None

    def project(self, query_vector: np.ndarray, phrase_vectors: np.ndarray, query_text: str = ""):
        """
        Project query embedding onto all capability subspaces.
        Returns ProjectionResult with active capabilities + scores.
        """
        all_prototypes = self._store.all_prototypes()
        if not all_prototypes:
            raise ProjectionError("No capability centroids loaded. Run: python -m setup.centroid_builder")

        # Tier 1: multi-prototype cosine scores
        tier1_scores = self._multi_prototype_scores(query_vector, all_prototypes)
        final_scores = tier1_scores.copy()
        method = "multi_prototype"

        # Tier 2: SetFit (if available and trained)
        if self._setfit_model is not None and query_text:
            try:
                tier2 = self._setfit_scores(query_text)
                if tier2:
                    final_scores = {
                        cap: 0.4 * tier1_scores.get(cap, 0.0) + 0.6 * tier2.get(cap, 0.0)
                        for cap in tier1_scores
                    }
                    method = "multi_prototype+setfit"
            except Exception:
                pass

        # Tier 3: cross-encoder for ambiguous queries
        ambiguous_resolved = False
        sorted_vals = sorted(final_scores.values(), reverse=True)
        if len(sorted_vals) >= 2 and (sorted_vals[0] - sorted_vals[1]) < settings.AMBIGUITY_THRESHOLD:
            if query_text:
                try:
                    self._ensure_cross_encoder()
                    if self._cross_encoder is not None:
                        ce = self._cross_encoder_scores(query_text)
                        if ce:
                            final_scores = {
                                cap: 0.3 * final_scores.get(cap, 0.0) + 0.7 * ce.get(cap, 0.0)
                                for cap in final_scores
                            }
                            ambiguous_resolved = True
                            method += "+cross_encoder"
                except Exception:
                    pass

        # Apply per-capability thresholds
        active = [cap for cap, s in final_scores.items() if s >= self._get_threshold(cap)]

        # Margin-based fallback: activate best if it's close to threshold and clearly leading
        if not active and final_scores:
            best_cap = max(final_scores, key=final_scores.get)
            best_s   = final_scores[best_cap]
            others   = sorted([v for k, v in final_scores.items() if k != best_cap], reverse=True)
            if best_s >= self._threshold - 0.08 and (not others or best_s > others[0] + 0.12):
                active = [best_cap]

        phrase_assignments = self._assign_phrases(phrase_vectors, all_prototypes)

        return ProjectionResult(
            active_capabilities=active,
            scores=final_scores,
            phrase_assignments=phrase_assignments,
            method_used=method,
            ambiguous_resolved=ambiguous_resolved,
        )

    def _multi_prototype_scores(self, query_vector, all_prototypes):
        scores = {}
        for cap_val, prototypes in all_prototypes.items():
            sims = prototypes @ query_vector
            scores[cap_val] = float(np.max(sims))
        return scores

    def _setfit_scores(self, query_text):
        if self._setfit_model is None:
            return {}
        try:
            probs = self._setfit_model.predict_proba([query_text])[0]
            labels = [cap.value for cap in Capability]
            return {labels[i]: float(probs[i]) for i in range(min(len(labels), len(probs)))}
        except Exception:
            return {}

    def _cross_encoder_scores(self, query_text):
        if self._cross_encoder is None:
            return {}
        hypotheses = {
            "gather":      "This text involves gathering information, research, or data collection.",
            "analyze":     "This text involves analyzing, understanding, or synthesizing concepts.",
            "plan":        "This text involves planning, designing, or structuring an approach.",
            "execute":     "This text involves building, implementing, or creating something.",
            "verify":      "This text involves testing, validating, or checking correctness.",
            "refine":      "This text involves improving, optimizing, or fixing existing work.",
            "communicate": "This text involves documenting, presenting, or communicating results.",
        }
        try:
            pairs  = [(query_text, hyp) for hyp in hypotheses.values()]
            raw    = self._cross_encoder.predict(pairs)
            keys   = list(hypotheses.keys())
            return {keys[i]: float(raw[i]) for i in range(len(keys))}
        except Exception:
            return {}

    def _assign_phrases(self, phrase_vectors, all_prototypes):
        if len(phrase_vectors) == 0:
            return {}
        assignments = {}
        for i, pv in enumerate(phrase_vectors):
            best_cap   = None
            best_score = -1.0
            for cap_val, protos in all_prototypes.items():
                score = float(np.max(protos @ pv))
                if score > best_score:
                    best_score = score
                    best_cap   = cap_val
            assignments[i] = best_cap
        return assignments


# ── Singleton ─────────────────────────────────────────────────────────────

_centroid_store  = None
_projector       = None


def get_projector():
    global _centroid_store, _projector
    if _projector is not None:
        return _projector
    _centroid_store = CentroidStore(str(settings.CENTROID_PATH))
    try:
        _centroid_store.load()
    except (CentroidNotFoundError, StaleCentroidError):
        pass
    _projector = CapabilityProjector(_centroid_store)
    _projector.load_setfit_if_available()
    return _projector
