"""
core/exceptions.py — Full exception hierarchy for NLGraph.
All modules raise from these — never bare Exception.
"""

from __future__ import annotations


# ── Base ───────────────────────────────────────────────────────────────────

class NLGraphError(Exception):
    """Root exception for all NLGraph errors."""


# ── Embedding ─────────────────────────────────────────────────────────────

class EmbeddingError(NLGraphError):
    """Raised when the embedding engine fails."""

class ModelNotCachedError(EmbeddingError):
    """Raised when LOCAL_FILES_ONLY=True but the model isn't in cache."""

class EmbedCacheInvalidatedError(EmbeddingError):
    """Raised when the embed cache is cleared due to model change."""


# ── Centroid / Setup ──────────────────────────────────────────────────────

class StaleCentroidError(NLGraphError):
    """Raised when loaded centroids were built with a different embed model."""

class CentroidNotFoundError(NLGraphError):
    """Raised when centroid file doesn't exist — run setup first."""

class CentroidBuildError(NLGraphError):
    """Raised when centroid building fails (e.g., insufficient seeds)."""


# ── Intent pipeline ───────────────────────────────────────────────────────

class PhraseSplitError(NLGraphError):
    """Raised when phrase splitting produces no usable phrases."""

class ProjectionError(NLGraphError):
    """Raised when capability projection produces an empty result."""

class SpacyModelError(NLGraphError):
    """Raised when the required spaCy model is not installed."""


# ── DAG ──────────────────────────────────────────────────────────────────

class DAGRepairFailedError(NLGraphError):
    """Raised when DAG validation fails after MAX_DAG_REPAIR_ATTEMPTS."""
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"DAG repair failed after max attempts. Errors: {errors}")

class CyclicDependencyError(NLGraphError):
    """Raised when a cycle is found and cannot be repaired."""


# ── Blackboard ────────────────────────────────────────────────────────────

class BlackboardCapacityError(NLGraphError):
    """Raised when FAISS fact index hits capacity and eviction fails."""

class BlackboardKeyError(NLGraphError):
    """Raised when an agent tries to read a key no upstream agent wrote."""


# ── Execution ────────────────────────────────────────────────────────────

class OllamaUnavailableError(NLGraphError):
    """Raised when Ollama service is unreachable."""

class OllamaModelMissingError(NLGraphError):
    """Raised when the configured model is not pulled in Ollama."""
    def __init__(self, model: str):
        self.model = model
        super().__init__(
            f"Model '{model}' not found in Ollama. "
            f"Run: ollama pull {model}"
        )

class AgentTimeoutError(NLGraphError):
    """Raised when an agent exceeds its execution timeout."""
    def __init__(self, agent_id: str, timeout: int):
        self.agent_id = agent_id
        super().__init__(f"Agent '{agent_id}' timed out after {timeout}s")

class AgentExecutionError(NLGraphError):
    """Raised when an agent fails during execution."""


# ── Preview / API ─────────────────────────────────────────────────────────

class PreviewExpiredError(NLGraphError):
    """Raised when a preview_id has expired (TTL exceeded)."""

class PreviewNotFoundError(NLGraphError):
    """Raised when a preview_id doesn't exist."""


# ── Memory ────────────────────────────────────────────────────────────────

class GlobalStoreError(NLGraphError):
    """Raised when Qdrant operations fail."""

class FeedbackError(NLGraphError):
    """Raised when feedback cannot be stored."""
